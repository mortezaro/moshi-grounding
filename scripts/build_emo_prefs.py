#!/usr/bin/env python
"""Build EMOTION-APPROPRIATENESS DPO pairs, on-policy, judge-labeled.

For each REAL emotional user turn (IEMOCAP hap/sad/ang/neu), generate diverse Moshi
responses (varying temperature AND Moshi's own emotion conditioning), then score each with
the judge (SER acoustic affect + Qwen LLM empathy/coherence/engagement). chosen = the reply
that best engages the user's emotion empathetically; rejected = generic / tone-deaf. We have
the user's audio, so pairs are PROPER stereo (ch1 = user turn, ch0 = Moshi response) — real
emotional context.

Two-phase (one GPU): Phase A generate+SER with Moshi, free it; Phase B LLM-judge + select +
write. Output = our DPO pref format (dpo_train.py consumes it).
"""
import sys, os, glob, re, json, argparse, collections, tempfile, gc
sys.path.insert(0, "/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import numpy as np, torch, sphn, pyarrow.parquet as pq
from moshi.models import loaders
from moshi.run_inference import InferenceState, get_condition_tensors
import ser_dim
EMAP = {"hap":"hap","exc":"hap","sad":"sad","ang":"ang","fru":"ang","neu":"neu"}

def empathetic_target(uv, ua):
    if uv < 0.42: return 0.58, 0.42
    if uv > 0.60: return 0.66, float(min(0.72, ua+0.05))
    return 0.55, 0.48
def affect_appropriateness(uvad, mvad):
    tv, ta = empathetic_target(uvad[2], uvad[0]); mv, ma = mvad[2], mvad[0]
    return float(np.clip(1 - np.sqrt((mv-tv)**2 + (0.6*(ma-ta))**2)/0.5, 0, 1))

def emo_contexts(sr, n_each, shard=0, nshards=1):
    fs = [f for f in sorted(glob.glob("/iopsstor/scratch/cscs/mrohania/hf_cache/datasets--Ar4ikov--iemocap_audio_text_splitted/snapshots/*/**/*.parquet", recursive=True)) if "test-" not in f]
    by = collections.defaultdict(list)
    for f in fs:
        t = pq.read_table(f); cols = t.column_names
        tc = "text" if "text" in cols else ("transcription" if "transcription" in cols else None)
        for r in t.select([c for c in ["titre","emotion","audio"]+([tc] if tc else []) if c in cols]).to_pylist():
            if "impro" not in (r.get("titre") or ""): continue
            e = EMAP.get(r.get("emotion"))
            if e and len(by[e]) < n_each: by[e].append((r["audio"]["bytes"], (r.get(tc) or "") if tc else ""))
        if all(len(by[e]) >= n_each for e in ["hap","sad","ang","neu"]): break
    out = []
    for e in ["hap","sad","ang","neu"]:
        for b, txt in by[e][:n_each][shard::nshards]:   # disjoint subset for this shard
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf: tf.write(b); tp = tf.name
            try: d, s = sphn.read(tp)
            finally: os.unlink(tp)
            w = d.mean(0) if d.ndim > 1 else d
            out.append((e, np.asarray(sphn.resample(w, s, sr) if s != sr else w, dtype=np.float32), txt))
    return out

def aligns_from_monologue(text_tokens, spm, pad_id, fr=12.5):
    words = []; cur = None
    for i, t in enumerate(text_tokens):
        ti = int(t)
        if ti < 0 or ti in (pad_id, spm.eos_id()): continue
        piece = spm.id_to_piece(ti); ts = i/fr
        if piece.startswith("▁") or cur is None:
            if cur: words.append(cur)
            cur = [piece.replace("▁",""), ts, ts+1/fr]
        else: cur[0] += piece; cur[2] = ts+1/fr
    if cur: words.append(cur)
    return [[w[0], [round(w[1],3), round(w[2],3)], "SPEAKER_MAIN"] for w in words if w[0]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n_each", type=int, default=12); ap.add_argument("--margin", type=float, default=0.18)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args(); dev = "cuda"; fr = 12.5
    ci = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",
                                             lora_weights=a.adapter, config_path=a.config)
    lm = ci.get_moshi(device=dev, dtype=torch.bfloat16, fuse_lora=True)
    mimi = ci.get_mimi(device=dev); spm = ci.get_text_tokenizer(); sr = mimi.sample_rate
    pad_id = getattr(lm, "text_padding_token_id", 3)
    proc, ser = ser_dim.load(dev)
    state = InferenceState(ci, mimi, spm, lm, 1, 1.0, dev, **ci.lm_gen_config)
    def ser_of(pcm):
        w16 = np.asarray(sphn.resample(pcm, sr, 16000), dtype=np.float32) if sr != 16000 else pcm
        return np.array(ser_dim.predict(proc, ser, dev, [w16]))[0]
    def decode(tt):
        ids = [int(t) for t in tt if int(t) >= 0 and int(t) not in (pad_id, spm.eos_id())]
        try: return spm.decode(ids).strip() if ids else ""
        except Exception: return ""
    CONDS = ["hap","neu","sad"]; TEMPS = [0.7, 0.9]
    ctxs = emo_contexts(sr, a.n_each, a.shard, a.nshards)
    contexts = []   # each: (true_e, user_audio, user_text, [candidates])
    print(f"[{a.label}] Phase A: generating candidates for {len(ctxs)} contexts", flush=True)
    for ci_, (true_e, uctx, utext) in enumerate(ctxs):
        uvad = ser_of(uctx); cands = []
        for cond in CONDS:
            os.environ["COND_EMOTION"] = cond
            state.lm_gen.condition_tensors = get_condition_tensors("moshi", lm, 1, 1.0)
            for tp in TEMPS:
                state.lm_gen.temp = tp; state.lm_gen.temp_text = tp
                state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
                inp = np.concatenate([uctx, np.zeros(int(6.0*sr), np.float32)])
                with torch.no_grad(): out = state.run(torch.from_numpy(inp[None, None]).to(dev))
                resp = out[0][1][0].detach().cpu().numpy().astype(np.float32); mtext = decode(out[0][0])
                fl = float((np.sqrt((resp[:len(resp)//480*480].reshape(-1,480)**2).mean(1)+1e-9) > 0.01).mean()) if len(resp) > 480 else 0.0
                mvad = ser_of(resp) if fl > 0.05 else np.array([0.5,0.5,0.5])
                cands.append(dict(audio=resp, text=mtext, ttoks=out[0][0], aff=affect_appropriateness(uvad, mvad)))
        contexts.append((true_e, uctx, utext or f"(a {true_e} user turn)", cands))
        if ci_ % 8 == 0: print(f"[{a.label}] ctx {ci_}/{len(ctxs)}", flush=True)
    del lm, mimi, state, ser, proc; gc.collect(); torch.cuda.empty_cache()
    # ---------- Phase B: LLM judge + select + write ----------
    import llm_judge
    ltok, lmodel = llm_judge.load(dev)
    ow = os.path.join(a.out, "data_stereo"); os.makedirs(ow, exist_ok=True)
    jo = open(os.path.join(a.out, f"prefs_{a.shard}.jsonl"), "w"); npair = 0
    print(f"[{a.label}] Phase B: judging + selecting", flush=True)
    for true_e, uctx, utext, cands in contexts:
        for c in cands:
            s = llm_judge.score(ltok, lmodel, dev, utext, c["text"])
            c["score"] = 0.35*s["empathy"] + 0.25*c["aff"] + 0.25*s["coherence"] + 0.15*s["engagement"]
            c["safe"] = s["safety"]
        cands = [c for c in cands if c["safe"] > 0.5 and len(c["text"]) > 0]
        if len(cands) < 2: continue
        cands.sort(key=lambda x: x["score"], reverse=True)
        best, worst = cands[0], cands[-1]
        if best["score"] - worst["score"] < a.margin: continue
        pid = f"{a.shard}_{npair}"; paths = {}
        for tag, c in (("c", best), ("r", worst)):
            m = c["audio"]; T = len(m)
            uch = np.zeros(T, dtype=np.float32); uch[:min(len(uctx), T)] = uctx[:T]
            buf = np.stack([m, uch]); pk = np.abs(buf).max()
            if pk > 0: buf = buf*(0.95/pk)
            bn = f"{pid}_{tag}"; sphn.write_wav(os.path.join(ow, bn+".wav"), buf, sr)
            json.dump({"alignments": aligns_from_monologue(c["ttoks"], spm, pad_id),
                       "text_conditions": {"emotion": "neu"}}, open(os.path.join(ow, bn+".json"), "w"))
            paths[tag] = (f"data_stereo/{bn}.wav", T/sr)
        jo.write(json.dumps({"chosen": paths["c"][0], "rejected": paths["r"][0],
                             "duration": max(paths["c"][1], paths["r"][1]),
                             "emotion": true_e, "margin": round(best["score"]-worst["score"],3)})+"\n")
        npair += 1
    jo.close(); print(f"[{a.label}] DONE emo-pref pairs={npair}", flush=True)

if __name__ == "__main__":
    main()
