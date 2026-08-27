#!/usr/bin/env python
"""Conversation-quality judge for Moshi fine-tunes.

Scores a model's *behavior* on two scenarios:
  (1) RESPONSE: real user turns (DailyTalk) -> does Moshi reply fluently, coherently,
      dynamically (asks questions / leads), with appropriate affect?
  (2) COLD-START: ~2s of silence -> does Moshi OPEN with a coherent greeting that steers
      the conversation, rather than random monologue?

Key trick: Moshi emits its own text inner-monologue, so we read the TRANSCRIPT of what it
said for free (no ASR) and score it. We also print every transcript — that directly
surfaces the "random stuff" problem.

Metrics (each 0..1, higher = better):
  fluency    speech-fraction of the audio (reuse of fluency_audio.speech_fraction)
  coherence  real-word ratio * (1 - repeated-bigram fraction)          [grounded/sane]
  dynamic    lexical diversity, blended with "asks a question" + length-in-range
  steering   asks a question or invites (?  / wh-word / "what about you")
  greeting   COLD-START only: opening is a short greeting/invite, not a ramble
  valence    SER valence of Moshi's delivery (context: is the affect alive, not flat)
  ------
  conv       composite conversational-quality score (weighted)

Usage:
  MOSHI_PY conv_judge.py --label L_long --adapter <p>/lora.safetensors --config <p>/config.json --n 12
  MOSHI_PY conv_judge.py --label BASE --n 12          # base moshiko (no adapter)
"""
import sys, os, glob, json, re, argparse
sys.path.insert(0, "/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import numpy as np, torch, sphn
from moshi.models import loaders
from moshi.run_inference import InferenceState
import ser_dim

GREET = re.compile(r"\b(hi|hey|hello|good (morning|afternoon|evening)|how are you|how's it going|"
                   r"what's up|nice to (meet|talk)|welcome|what would you like|what do you want to talk)\b", re.I)
WH = re.compile(r"\b(what|why|how|when|where|who|which|do you|are you|would you|tell me|what about you)\b", re.I)

def speech_fraction(pcm, sr, hop=0.02, thr=0.01):
    n = int(hop*sr); m = len(pcm)//n
    if m == 0: return 0.0
    fr = np.sqrt((pcm[:m*n].reshape(m, n)**2).mean(1)+1e-9)
    return float((fr > thr).mean())

def decode_text(text_tokens, tok, pad_id):
    ids = [int(t) for t in text_tokens if int(t) >= 0 and int(t) not in (pad_id, tok.eos_id())]
    if not ids: return ""
    try: return tok.decode(ids).strip()
    except Exception: return ""

def text_metrics(txt, cold):
    words = re.findall(r"[a-zA-Z']+", txt.lower())
    n = len(words)
    if n == 0:
        return dict(coherence=0.0, dynamic=0.0, steering=0.0, greeting=0.0, nwords=0)
    real = sum(1 for w in words if len(w) > 1 or w in ("a", "i")) / n
    bigrams = list(zip(words, words[1:]))
    rep = 1.0 - (len(set(bigrams))/len(bigrams)) if bigrams else 0.0
    coherence = max(0.0, real * (1.0 - rep))
    diversity = len(set(words))/n
    length_ok = 1.0 if 3 <= n <= 60 else (0.4 if n < 3 else 0.5)
    asks = 1.0 if ("?" in txt or WH.search(txt)) else 0.0
    dynamic = float(np.clip(0.5*diversity + 0.3*length_ok + 0.2*asks, 0, 1))
    steering = float(np.clip(0.6*asks + 0.4*(1.0 if WH.search(txt) else 0.0), 0, 1))
    greeting = 0.0
    if cold:
        greeting = float(np.clip((1.0 if GREET.search(txt) else 0.0)*0.7
                                 + (0.3 if 2 <= n <= 20 else 0.0), 0, 1))
    return dict(coherence=coherence, dynamic=dynamic, steering=steering, greeting=greeting, nwords=n)

def pick_user_turns(mimi, n):
    """Real user turns from DailyTalk (audio + transcript for context)."""
    base = "/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkContiguous/data_stereo"
    out = []
    for wf in sorted(glob.glob(base+"/*.wav"))[-200:]:
        try: al = json.load(open(wf[:-4]+".json"))["alignments"]
        except Exception: continue
        d, _ = sphn.read(wf, sample_rate=mimi.sample_rate)
        cur = None; segs = []
        for w, ts, s in al:
            if cur is None: cur = [ts[0], ts[1], [w]]
            elif ts[0]-cur[1] <= 0.6: cur[1] = ts[1]; cur[2].append(w)
            else: segs.append(cur); cur = [ts[0], ts[1], [w]]
        if cur: segs.append(cur)
        for t0, t1, ws in segs:
            if (t1-t0) > 1.5:
                seg = d[1][int(t0*mimi.sample_rate):int(t1*mimi.sample_rate)]
                if np.sqrt((seg**2).mean()+1e-9) > 0.02:
                    out.append((os.path.basename(wf), seg, " ".join(ws)))
            if len(out) >= n: return out
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=""); ap.add_argument("--config", default="")
    ap.add_argument("--label", required=True); ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cfg", type=float, default=1.0)
    a = ap.parse_args(); dev = "cuda"
    if a.adapter:
        ci = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",
                                                 lora_weights=a.adapter, config_path=a.config)
        lm = ci.get_moshi(device=dev, dtype=torch.bfloat16, fuse_lora=True)
    else:
        ci = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        lm = ci.get_moshi(device=dev, dtype=torch.bfloat16)
    mimi = ci.get_mimi(device=dev); tok = ci.get_text_tokenizer(); sr = mimi.sample_rate
    proc, ser = ser_dim.load(dev)
    pad_id = getattr(lm, "text_padding_token_id", 3)
    os.environ.setdefault("COND_EMOTION", "neu")
    state = InferenceState(ci, mimi, tok, lm, 1, a.cfg, dev, **ci.lm_gen_config)
    outdir = "/iopsstor/scratch/cscs/mrohania/audio_samples/judge"; os.makedirs(outdir, exist_ok=True)

    def run_one(user_pcm):
        state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
        out = state.run(torch.from_numpy(user_pcm[None, None]).to(dev))
        text_tokens = out[0][0]; audio = out[0][1][0].detach().cpu().numpy().astype(np.float32)
        txt = decode_text(text_tokens, tok, pad_id)
        return audio, txt

    scenarios = []
    # cold-start: 2s silence -> test the opening/greeting
    scenarios.append(("COLDSTART", np.zeros(int(2.0*sr), dtype=np.float32), "", True))
    for fn, seg, ctx in pick_user_turns(mimi, a.n):
        scenarios.append((fn, np.asarray(seg, dtype=np.float32), ctx, False))

    rows = []
    print(f"\n===== JUDGE  model={a.label}  cfg={a.cfg} =====")
    for i, (name, upcm, ctx, cold) in enumerate(scenarios):
        audio, txt = run_one(upcm)
        fl = speech_fraction(audio, sr)
        try:
            w16 = np.asarray(sphn.resample(audio, sr, 16000), dtype=np.float32)
            val = float(np.array(ser_dim.predict(proc, ser, dev, [w16]))[0][2]) if fl > 0.05 else 0.5
        except Exception: val = 0.5
        tm = text_metrics(txt, cold)
        rows.append((cold, fl, tm["coherence"], tm["dynamic"], tm["steering"], tm["greeting"], val, tm["nwords"]))
        tag = "COLD " if cold else "resp "
        print(f"[{tag}{i:02d}] fl={fl:.2f} coh={tm['coherence']:.2f} dyn={tm['dynamic']:.2f} "
              f"steer={tm['steering']:.2f} greet={tm['greeting']:.2f} val={val:.2f} nw={tm['nwords']:>2}  "
              f"| MOSHI: {txt[:90]!r}")
        if i < 4:
            sphn.write_wav(f"{outdir}/{a.label}_{i:02d}_{name}.wav", audio[None], sr)

    arr = np.array([r[1:] for r in rows], dtype=float)  # fl,coh,dyn,steer,greet,val,nw
    resp = arr[[not r[0] for r in rows]]; cold = arr[[r[0] for r in rows]]
    def m(a_, j): return float(a_[:, j].mean()) if len(a_) else 0.0
    fl, coh, dyn, steer, val = m(resp,0), m(resp,1), m(resp,2), m(resp,3), m(resp,5)
    greet = m(cold, 4)
    conv = float(np.clip(0.30*fl + 0.25*coh + 0.20*dyn + 0.15*steer + 0.10*greet, 0, 1))
    print(f"\n--- {a.label} SUMMARY (n_resp={len(resp)}) ---")
    print(f"fluency={fl:.3f} coherence={coh:.3f} dynamic={dyn:.3f} steering={steer:.3f} "
          f"greeting={greet:.3f} valence={val:.3f}")
    print(f"CONV_QUALITY={conv:.3f}")
    json.dump(dict(label=a.label, fluency=fl, coherence=coh, dynamic=dyn, steering=steer,
                   greeting=greet, valence=val, conv=conv),
              open(f"{outdir}/{a.label}_scores.json", "w"), indent=2)

if __name__ == "__main__":
    main()
