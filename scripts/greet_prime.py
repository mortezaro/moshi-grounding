#!/usr/bin/env python
"""Greeting-prime prototype: make Moshi OPEN with a chosen starter in its own voice.

Mechanism: Moshi already talks into silence — we don't force *when* it speaks, we steer
*what* it says. Via on_text_hook we substitute the starter's tokens onto Moshi's own
word-frames (leaving padding/pause frames alone, so rhythm stays natural), then release to
free generation once the starter is done. Rotates a few starters.

This validates the mechanism offline (prints transcript, saves audio to listen). If good,
the same hook wires into reactive_server for a coherent, dynamic opening.
"""
import sys, os, re, argparse
sys.path.insert(0, "/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import numpy as np, torch, sphn
from moshi.models import loaders
from moshi.run_inference import InferenceState

STARTERS = [
    "Hey! Good to talk to you. What's on your mind today?",
    "Hi there! How has your day been so far?",
    "Hello! What would you like to talk about?",
    "Hey, welcome. Is there something you've been thinking about lately?",
    "Hi! I'm all ears. What's going on with you today?",
]

def speech_fraction(pcm, sr, hop=0.02, thr=0.01):
    n = int(hop*sr); m = len(pcm)//n
    if m == 0: return 0.0
    fr = np.sqrt((pcm[:m*n].reshape(m, n)**2).mean(1)+1e-9)
    return float((fr > thr).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=""); ap.add_argument("--config", default="")
    ap.add_argument("--label", required=True); ap.add_argument("--secs", type=float, default=7.0)
    a = ap.parse_args(); dev = "cuda"
    if a.adapter:
        ci = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",
                                                 lora_weights=a.adapter, config_path=a.config)
        lm = ci.get_moshi(device=dev, dtype=torch.bfloat16, fuse_lora=True)
    else:
        ci = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        lm = ci.get_moshi(device=dev, dtype=torch.bfloat16)
    mimi = ci.get_mimi(device=dev); tok = ci.get_text_tokenizer(); sr = mimi.sample_rate
    pad_id = getattr(lm, "text_padding_token_id", 3); eos = tok.eos_id()
    os.environ.setdefault("COND_EMOTION", "neu")
    state = InferenceState(ci, mimi, tok, lm, 1, 1.0, dev, **ci.lm_gen_config)
    outdir = "/iopsstor/scratch/cscs/mrohania/audio_samples/greet"; os.makedirs(outdir, exist_ok=True)

    def decode_text(tt_list):
        ids = [int(t) for t in tt_list if int(t) >= 0 and int(t) not in (pad_id, eos)]
        try: return tok.decode(ids).strip() if ids else ""
        except Exception: return ""

    print(f"\n===== GREETING-PRIME  model={a.label} =====")
    for si, starter in enumerate(STARTERS):
        target = tok.encode(starter)                 # subword ids of the starter
        ptr = {"i": 0}
        def hook(tt, target=target, ptr=ptr):
            if ptr["i"] >= len(target): return None  # done -> release to free generation
            v = int(tt.reshape(-1)[0])
            if v == pad_id or v == eos: return None   # keep natural pauses
            rep = torch.full_like(tt, target[ptr["i"]]); ptr["i"] += 1
            return rep
        state.lm_gen.on_text_hook = hook
        state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
        sil = np.zeros(int(a.secs*sr), dtype=np.float32)
        out = state.run(torch.from_numpy(sil[None, None]).to(dev))
        txt = decode_text(out[0][0]); audio = out[0][1][0].detach().cpu().numpy().astype(np.float32)
        state.lm_gen.on_text_hook = None
        fl = speech_fraction(audio, sr)
        forced = ptr["i"]; total = len(target)
        print(f"[{si}] forced {forced}/{total} tokens  fl={fl:.2f}")
        print(f"     WANT : {starter!r}")
        print(f"     MOSHI: {txt[:110]!r}")
        sphn.write_wav(f"{outdir}/{a.label}_starter{si}.wav", audio[None], sr)
    print(f"\nwavs saved to {outdir}/  (listen to check the opening is intelligible)")

if __name__ == "__main__":
    main()
