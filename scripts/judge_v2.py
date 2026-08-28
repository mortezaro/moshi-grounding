#!/usr/bin/env python
"""Multi-dimensional judge for an emotion-aware companion — STEP 1 (affect core).

Scores a model on REAL emotional contexts (IEMOCAP): for each user clip, read the user's
affect (SER), generate Moshi's response, read Moshi's affect + monologue text, and score:

  affect_appropriateness  ⭐ empathetic fit: user down -> Moshi warm/gentle (NOT matching-sad),
                             user up -> mirror positive. SER(user) -> target; SER(moshi) -> proximity.
  coherence   real-word ratio * (1 - repeated-bigram)        [heuristic; -> LLM in step 2]
  engagement  lexical diversity + asks-a-question + length
  quality     acoustic: speech-fraction, not-too-much dead air
  safety      keyword flag                                    [heuristic; -> LLM in step 2]
  ------
  companion   composite (affect-weighted)

Step 2 will replace coherence/engagement/safety with an LLM judge (Qwen2.5-7B-Instruct).
"""
import sys, os, glob, re, argparse, collections
sys.path.insert(0, "/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import numpy as np, torch, sphn, tempfile, pyarrow.parquet as pq
from moshi.models import loaders
from moshi.run_inference import InferenceState
import ser_dim

WH = re.compile(r"\b(what|why|how|when|where|who|which|do you|are you|would you|tell me|what about you)\b", re.I)
UNSAFE = re.compile(r"\b(kill yourself|suicide|self[- ]?injury|self[- ]?harm|hate you|worthless|die)\b", re.I)
EMAP = {"hap":"hap","exc":"hap","sad":"sad","ang":"ang","fru":"ang","neu":"neu"}

def speech_fraction(pcm, sr, hop=0.02, thr=0.01):
    n = int(hop*sr); m = len(pcm)//n
    if m == 0: return 0.0
    fr = np.sqrt((pcm[:m*n].reshape(m, n)**2).mean(1)+1e-9)
    return float((fr > thr).mean())

def empathetic_target(user_val, user_aro):
    """Companion policy: support negative affect (warm, not sad), mirror positive."""
    if user_val < 0.42:   return 0.58, 0.42                    # user down -> warm + calm
    if user_val > 0.60:   return 0.66, float(min(0.72, user_aro+0.05))  # user up -> mirror
    return 0.55, 0.48                                          # neutral -> gently warm

def affect_appropriateness(user_vad, moshi_vad):
    tv, ta = empathetic_target(user_vad[2], user_vad[0])
    mv, ma = moshi_vad[2], moshi_vad[0]
    d = np.sqrt((mv-tv)**2 + (0.6*(ma-ta))**2)                 # valence weighted over arousal
    return float(np.clip(1 - d/0.5, 0, 1))

def text_scores(txt):
    words = re.findall(r"[a-zA-Z']+", txt.lower()); n = len(words)
    if n == 0: return dict(coherence=0.0, engagement=0.0, safety=1.0, nwords=0)
    bg = list(zip(words, words[1:])); rep = 1.0-(len(set(bg))/len(bg)) if bg else 0.0
    coherence = max(0.0, 1.0-rep)
    diversity = len(set(words))/n
    asks = 1.0 if ("?" in txt or WH.search(txt)) else 0.0
    length_ok = 1.0 if 3 <= n <= 45 else (0.3 if n > 45 else 0.5)
    engagement = float(np.clip(0.5*diversity + 0.3*length_ok + 0.2*asks, 0, 1))
    safety = 0.0 if UNSAFE.search(txt) else 1.0
    return dict(coherence=coherence, engagement=engagement, safety=safety, nwords=n)

def emo_contexts(sr, n_each=4):
    fs = [f for f in sorted(glob.glob("/iopsstor/scratch/cscs/mrohania/hf_cache/datasets--Ar4ikov--iemocap_audio_text_splitted/snapshots/*/**/*.parquet", recursive=True)) if "test-" not in f]
    by = collections.defaultdict(list)
    for f in fs:
        for r in pq.read_table(f, columns=["titre","emotion","audio"]).to_pylist():
            if "impro" not in (r["titre"] or ""): continue
            e = EMAP.get(r["emotion"])
            if e and len(by[e]) < n_each: by[e].append(r["audio"]["bytes"])
        if all(len(by[e]) >= n_each for e in ["hap","sad","ang","neu"]): break
    out = []
    for e in ["hap","sad","ang","neu"]:
        for b in by[e][:n_each]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf: tf.write(b); tp = tf.name
            try: d, s = sphn.read(tp)
            finally: os.unlink(tp)
            w = d.mean(0) if d.ndim > 1 else d
            out.append((e, np.asarray(sphn.resample(w, s, sr) if s != sr else w, dtype=np.float32)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--n_each", type=int, default=4)
    a = ap.parse_args(); dev = "cuda"
    ci = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",
                                             lora_weights=a.adapter, config_path=a.config)
    lm = ci.get_moshi(device=dev, dtype=torch.bfloat16, fuse_lora=True)
    mimi = ci.get_mimi(device=dev); spm = ci.get_text_tokenizer(); sr = mimi.sample_rate
    pad_id = getattr(lm, "text_padding_token_id", 3)
    proc, ser = ser_dim.load(dev)
    os.environ.setdefault("COND_EMOTION", "neu")
    state = InferenceState(ci, mimi, spm, lm, 1, 1.0, dev, **ci.lm_gen_config)

    def ser_of(pcm):
        w16 = np.asarray(sphn.resample(pcm, sr, 16000), dtype=np.float32) if sr != 16000 else pcm
        return np.array(ser_dim.predict(proc, ser, dev, [w16]))[0]   # [aro,dom,val]
    def decode(tt):
        ids = [int(t) for t in tt if int(t) >= 0 and int(t) not in (pad_id, spm.eos_id())]
        try: return spm.decode(ids).strip() if ids else ""
        except Exception: return ""

    print(f"\n===== JUDGE v2  {a.label} =====")
    agg = collections.defaultdict(list)
    for true_e, uctx in emo_contexts(sr, a.n_each):
        uvad = ser_of(uctx)
        state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
        inp = np.concatenate([uctx, np.zeros(int(6.0*sr), np.float32)])
        with torch.no_grad(): out = state.run(torch.from_numpy(inp[None, None]).to(dev))
        resp = out[0][1][0].detach().cpu().numpy().astype(np.float32); txt = decode(out[0][0])
        fl = speech_fraction(resp, sr)
        mvad = ser_of(resp) if fl > 0.05 else np.array([0.5, 0.5, 0.5])
        aff = affect_appropriateness(uvad, mvad)
        ts = text_scores(txt)
        quality = float(np.clip(0.5 + (fl-0.3), 0, 1)) if fl < 0.6 else float(np.clip(1.0-(fl-0.6), 0.4, 1))
        for k, v in dict(affect=aff, coherence=ts["coherence"], engagement=ts["engagement"],
                         quality=quality, safety=ts["safety"]).items(): agg[k].append(v)
        print(f"[user {true_e:>3} val={uvad[2]:.2f}] moshi val={mvad[2]:.2f} aff={aff:.2f} "
              f"coh={ts['coherence']:.2f} eng={ts['engagement']:.2f} q={quality:.2f} sfe={ts['safety']:.0f} "
              f"| {txt[:70]!r}")
    m = {k: float(np.mean(v)) for k, v in agg.items()}
    companion = float(np.clip(0.35*m["affect"] + 0.25*m["coherence"] + 0.20*m["engagement"]
                              + 0.15*m["quality"] + 0.05*m["safety"], 0, 1))
    print(f"\n--- {a.label} --- affect={m['affect']:.3f} coherence={m['coherence']:.3f} "
          f"engagement={m['engagement']:.3f} quality={m['quality']:.3f} safety={m['safety']:.3f}")
    print(f"COMPANION_SCORE={companion:.3f}")

if __name__ == "__main__":
    main()
