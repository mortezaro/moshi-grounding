#!/usr/bin/env python
"""Generate on-policy DPO preference pairs for a Moshi model.

For each context (cold-start silence AND real user turns), sample K responses at varied
temperatures, score each with the conversation judge (coherence + steering + dynamic, minus
an over-talk/babble penalty), and emit the best as CHOSEN and worst as REJECTED when the
quality gap is clear. Pairs are written in the trainer's stereo-wav + alignment format
(ch0 = Moshi response, ch1 = user context), so dpo_train.py reuses the interleaver.

Output under --out:  data_stereo/<id>_{c,r}.wav + .json ,  and prefs_<shard>.jsonl with
lines {"chosen","rejected","duration"}.
"""
import sys, os, glob, json, re, argparse, collections
sys.path.insert(0, "/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import numpy as np, torch, sphn
from moshi.models import loaders
from moshi.run_inference import InferenceState

WH = re.compile(r"\b(what|why|how|when|where|who|which|do you|are you|would you|tell me|what about you)\b", re.I)

def score_text(txt, fl):
    """Higher = better conversational turn. Penalize babble (much talk, low diversity)."""
    words = re.findall(r"[a-zA-Z']+", txt.lower()); n = len(words)
    if n == 0: return 0.0, 0
    bg = list(zip(words, words[1:]))
    rep = 1.0 - (len(set(bg))/len(bg)) if bg else 0.0
    diversity = len(set(words))/n
    coherence = max(0.0, (1.0 - rep))
    asks = 1.0 if ("?" in txt or WH.search(txt)) else 0.0
    length_ok = 1.0 if 3 <= n <= 40 else (0.3 if n > 40 else 0.5)
    # babble penalty: lots of speech-frames but low diversity / very long => talking nonsense
    babble = max(0.0, fl - 0.45) * (1.0 - diversity) + (0.3 if n > 45 else 0.0)
    s = 0.35*coherence + 0.25*diversity + 0.20*length_ok + 0.20*asks - 0.5*babble
    return float(s), n

def speech_fraction(pcm, sr, hop=0.02, thr=0.01):
    nn = int(hop*sr); m = len(pcm)//nn
    if m == 0: return 0.0
    fr = np.sqrt((pcm[:m*nn].reshape(m, nn)**2).mean(1)+1e-9)
    return float((fr > thr).mean())

def pick_user_turns(mimi, n):
    base = "/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkContiguous/data_stereo"
    out = []
    for wf in sorted(glob.glob(base+"/*.wav"))[-260:]:
        try: al = json.load(open(wf[:-4]+".json"))["alignments"]
        except Exception: continue
        d, _ = sphn.read(wf, sample_rate=mimi.sample_rate)
        cur = None; segs = []
        for w, ts, s in al:
            if cur is None: cur = [ts[0], ts[1]]
            elif ts[0]-cur[1] <= 0.6: cur[1] = ts[1]
            else: segs.append(cur); cur = [ts[0], ts[1]]
        if cur: segs.append(cur)
        for t0, t1 in segs:
            if (t1-t0) > 1.5:
                seg = d[1][int(t0*mimi.sample_rate):int(t1*mimi.sample_rate)]
                if np.sqrt((seg**2).mean()+1e-9) > 0.02: out.append(np.asarray(seg, dtype=np.float32))
            if len(out) >= n: return out
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--gap", type=float, default=0.25); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--resp_secs", type=float, default=7.0)
    a = ap.parse_args(); dev = "cuda"; fr = 12.5
    ci = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",
                                             lora_weights=a.adapter, config_path=a.config)
    lm = ci.get_moshi(device=dev, dtype=torch.bfloat16, fuse_lora=True)
    mimi = ci.get_mimi(device=dev); tok = ci.get_text_tokenizer(); sr = mimi.sample_rate
    pad_id = getattr(lm, "text_padding_token_id", 3); eos = tok.eos_id()
    os.environ.setdefault("COND_EMOTION", "neu")
    cfg = dict(ci.lm_gen_config)
    ow = os.path.join(a.out, "data_stereo"); os.makedirs(ow, exist_ok=True)
    jo = open(os.path.join(a.out, f"prefs_{a.shard}.jsonl"), "w")

    def align_from_monologue(text_tokens):
        words = []; cur = None
        for i, t in enumerate(text_tokens):
            ti = int(t)
            if ti < 0 or ti in (pad_id, eos): continue
            piece = tok.id_to_piece(ti); t_s = i/fr
            if piece.startswith("▁") or cur is None:
                if cur: words.append(cur)
                cur = [piece.replace("▁", ""), t_s, t_s + 1/fr]
            else:
                cur[0] += piece; cur[2] = t_s + 1/fr
        if cur: words.append(cur)
        return [[w[0], [round(w[1], 3), round(w[2], 3)], "SPEAKER_MAIN"] for w in words if w[0]]

    def gen(state, user_pcm, temp_text, temp):
        state.lm_gen.temp_text = temp_text; state.lm_gen.temp = temp
        state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
        inp = np.concatenate([user_pcm, np.zeros(int(a.resp_secs*sr), dtype=np.float32)])
        out = state.run(torch.from_numpy(inp[None, None]).to(dev))
        audio = out[0][1][0].detach().cpu().numpy().astype(np.float32)
        return audio, out[0][0]

    state = InferenceState(ci, mimi, tok, lm, 1, 1.0, dev, **cfg)
    contexts = [("cold", np.zeros(int(0.5*sr), dtype=np.float32)) for _ in range(a.n//3)]
    contexts += [("resp", u) for u in pick_user_turns(mimi, a.n - a.n//3)]
    temps = [(0.5, 0.5), (0.8, 0.7), (1.1, 0.9), (1.4, 1.1)][:a.k]  # wide spread -> clear pairs

    npair = 0
    for ci_, (kind, uctx) in enumerate(contexts):
        cands = []
        for (tt, ta) in temps:
            audio, ttoks = gen(state, uctx, tt, ta)
            fl = speech_fraction(audio, sr); s, nw = score_text(_decode(ttoks, tok, pad_id, eos), fl)
            cands.append((s, audio, ttoks, nw))
        cands.sort(key=lambda x: x[0], reverse=True)
        best, worst = cands[0], cands[-1]
        if best[0] - worst[0] < 0.15 or best[3] < 2:   # need a clear, non-empty preference
            continue
        pair_id = f"{a.shard}_{npair}"
        paths = {}
        for tag, cand in (("c", best), ("r", worst)):
            audio = cand[1]; total = len(audio)
            uch = np.zeros(total, dtype=np.float32); uch[:min(len(uctx), total)] = uctx[:total]
            buf = np.stack([audio, uch]); pk = np.abs(buf).max()
            if pk > 0: buf = buf*(0.95/pk)
            bn = f"{pair_id}_{tag}"
            sphn.write_wav(os.path.join(ow, bn+".wav"), buf, sr)
            json.dump({"alignments": align_from_monologue(cand[2]),
                       "text_conditions": {"emotion": os.environ["COND_EMOTION"]}},
                      open(os.path.join(ow, bn+".json"), "w"))
            paths[tag] = (f"data_stereo/{bn}.wav", total/sr)
        jo.write(json.dumps({"chosen": paths["c"][0], "rejected": paths["r"][0],
                             "duration": max(paths["c"][1], paths["r"][1]),
                             "gap": round(best[0]-worst[0], 3), "kind": kind})+"\n")
        npair += 1
        if ci_ % 10 == 0: print(f"[{a.label} shard{a.shard}] ctx {ci_}/{len(contexts)} pairs={npair}", flush=True)
    jo.close()
    print(f"[{a.label} shard{a.shard}] DONE pairs={npair}", flush=True)

def _decode(text_tokens, tok, pad_id, eos):
    ids = [int(t) for t in text_tokens if int(t) >= 0 and int(t) not in (pad_id, eos)]
    try: return tok.decode(ids).strip() if ids else ""
    except Exception: return ""

if __name__ == "__main__":
    main()
