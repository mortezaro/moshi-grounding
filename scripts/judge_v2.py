#!/usr/bin/env python
"""Multi-dimensional judge for an emotion-aware companion (integrated: SER + LLM).

On REAL emotional contexts (IEMOCAP): for each user clip, read user affect (SER), generate
Moshi's response, read Moshi affect (SER) + monologue text. Then score:

  affect  ⭐ ACOUSTIC empathy: does Moshi SOUND right for the user? user down -> warm/gentle,
             user up -> mirror. SER(user)->target; SER(moshi)->proximity.
  coherence / engagement / empathy / safety : LLM judge (Qwen2.5-7B) on the WORDS.
  ------
  companion : composite (acoustic affect + semantic empathy both weighted).

Two-phase in one job: generate with Moshi (+SER), free it, then load the LLM and score.
"""
import sys, os, glob, re, argparse, collections, tempfile, gc
sys.path.insert(0, "/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import numpy as np, torch, sphn, pyarrow.parquet as pq
from moshi.models import loaders
from moshi.run_inference import InferenceState
import ser_dim
EMAP = {"hap":"hap","exc":"hap","sad":"sad","ang":"ang","fru":"ang","neu":"neu"}

def empathetic_target(user_val, user_aro):
    if user_val < 0.42:   return 0.58, 0.42
    if user_val > 0.60:   return 0.66, float(min(0.72, user_aro+0.05))
    return 0.55, 0.48
def affect_appropriateness(uvad, mvad):
    tv, ta = empathetic_target(uvad[2], uvad[0]); mv, ma = mvad[2], mvad[0]
    return float(np.clip(1 - np.sqrt((mv-tv)**2 + (0.6*(ma-ta))**2)/0.5, 0, 1))

def emo_contexts(sr, n_each):
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
        for b, txt in by[e][:n_each]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf: tf.write(b); tp = tf.name
            try: d, s = sphn.read(tp)
            finally: os.unlink(tp)
            w = d.mean(0) if d.ndim > 1 else d
            out.append((e, np.asarray(sphn.resample(w, s, sr) if s != sr else w, dtype=np.float32), txt))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--n_each", type=int, default=4)
    ap.add_argument("--dpo", default="", help="optional dpo_lora.safetensors to stack on the fused fine-tune")
    a = ap.parse_args(); dev = "cuda"
    # ---------- Phase A: generate with Moshi + read affect ----------
    ci = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",
                                             lora_weights=a.adapter, config_path=a.config)
    lm = ci.get_moshi(device=dev, dtype=torch.bfloat16, fuse_lora=True)
    if a.dpo:
        from moshi.modules.lora import replace_all_linear_with_lora, LoRALinear
        from safetensors.torch import load_file
        replace_all_linear_with_lora(lm, 16, scaling=1.0, device=dev, dtype=torch.bfloat16)
        for m in lm.modules():
            if isinstance(m, LoRALinear): torch.nn.init.zeros_(m.lora_B.weight)
        lm.load_state_dict(load_file(a.dpo), strict=False)
    mimi = ci.get_mimi(device=dev); spm = ci.get_text_tokenizer(); sr = mimi.sample_rate
    pad_id = getattr(lm, "text_padding_token_id", 3)
    proc, ser = ser_dim.load(dev)
    os.environ.setdefault("COND_EMOTION", "neu")
    state = InferenceState(ci, mimi, spm, lm, 1, 1.0, dev, **ci.lm_gen_config)
    def ser_of(pcm):
        w16 = np.asarray(sphn.resample(pcm, sr, 16000), dtype=np.float32) if sr != 16000 else pcm
        return np.array(ser_dim.predict(proc, ser, dev, [w16]))[0]
    def decode(tt):
        ids = [int(t) for t in tt if int(t) >= 0 and int(t) not in (pad_id, spm.eos_id())]
        try: return spm.decode(ids).strip() if ids else ""
        except Exception: return ""
    rows = []
    for true_e, uctx, utext in emo_contexts(sr, a.n_each):
        uvad = ser_of(uctx)
        state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
        inp = np.concatenate([uctx, np.zeros(int(6.0*sr), np.float32)])
        with torch.no_grad(): out = state.run(torch.from_numpy(inp[None, None]).to(dev))
        resp = out[0][1][0].detach().cpu().numpy().astype(np.float32); mtext = decode(out[0][0])
        fl = float((np.sqrt((resp.reshape(-1, int(0.02*sr))[:len(resp)//int(0.02*sr)]**2).mean(1)+1e-9) > 0.01).mean()) if len(resp) > int(0.02*sr) else 0.0
        mvad = ser_of(resp) if fl > 0.05 else np.array([0.5, 0.5, 0.5])
        rows.append(dict(true_e=true_e, utext=utext or f"(a {true_e} user turn)", mtext=mtext,
                         aff=affect_appropriateness(uvad, mvad), uval=uvad[2], mval=mvad[2]))
    del lm, mimi, state, ser, proc; gc.collect(); torch.cuda.empty_cache()
    # ---------- Phase B: LLM judge the words ----------
    import llm_judge
    ltok, lmodel = llm_judge.load(dev)
    print(f"\n===== JUDGE v2 (SER+LLM)  {a.label} =====")
    agg = collections.defaultdict(list)
    for r in rows:
        s = llm_judge.score(ltok, lmodel, dev, r["utext"], r["mtext"])
        for k, v in dict(affect=r["aff"], coherence=s["coherence"], engagement=s["engagement"],
                         empathy=s["empathy"], safety=s["safety"]).items(): agg[k].append(v)
        print(f"[{r['true_e']:>3} uval={r['uval']:.2f}] aff={r['aff']:.2f} coh={s['coherence']:.2f} "
              f"eng={s['engagement']:.2f} emp={s['empathy']:.2f} safe={s['safety']:.0f} | {r['mtext'][:60]!r}")
    m = {k: float(np.mean(v)) for k, v in agg.items()}
    companion = float(np.clip(0.25*m["affect"] + 0.25*m["empathy"] + 0.25*m["coherence"]
                              + 0.15*m["engagement"] + 0.10*m["safety"], 0, 1))
    print(f"\n--- {a.label} --- affect(acoustic)={m['affect']:.3f} empathy(text)={m['empathy']:.3f} "
          f"coherence={m['coherence']:.3f} engagement={m['engagement']:.3f} safety={m['safety']:.3f}")
    print(f"COMPANION_SCORE={companion:.3f}")

if __name__ == "__main__":
    main()
