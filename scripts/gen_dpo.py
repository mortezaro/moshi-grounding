#!/usr/bin/env python
"""Generate BEFORE vs AFTER audio for a DPO winner, to confirm by ear.

Loads base+adapter (fused) + the fresh DPO LoRA; BEFORE = same model with DPO LoRA
scaling=0 (i.e. the original fine-tune), AFTER = scaling on (DPO'd). Same model, exact A/B.
Scenarios: cold-start (opening/greeting) + real user turns (response). Saves wavs + prints
the monologue transcript for each.
"""
import sys, os, glob, json, argparse
sys.path.insert(0, "/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import numpy as np, torch, sphn
from moshi.models import loaders
from moshi.modules.lora import replace_all_linear_with_lora, LoRALinear
from moshi.run_inference import InferenceState, get_condition_tensors
from safetensors.torch import load_file

def set_scaling(model, val):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if not hasattr(m, "_bs"): m._bs = m.scaling
            m.scaling = m._bs if val is None else val

def pick_user_turns(mimi, n):
    base = "/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkContiguous/data_stereo"
    out = []
    for wf in sorted(glob.glob(base+"/*.wav"))[-120:]:
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
    ap.add_argument("--dpo", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--rank", type=int, default=16)
    a = ap.parse_args(); dev = "cuda"
    ci = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",
                                             lora_weights=a.adapter, config_path=a.config)
    lm = ci.get_moshi(device=dev, dtype=torch.bfloat16, fuse_lora=True)
    mimi = ci.get_mimi(device=dev); spm = ci.get_text_tokenizer(); sr = mimi.sample_rate
    pad_id = getattr(lm, "text_padding_token_id", 3)
    replace_all_linear_with_lora(lm, a.rank, scaling=1.0, device=dev, dtype=torch.bfloat16)
    for m in lm.modules():
        if isinstance(m, LoRALinear): torch.nn.init.zeros_(m.lora_B.weight)
    missing = lm.load_state_dict(load_file(a.dpo), strict=False)
    lm.eval()
    os.environ.setdefault("COND_EMOTION", "neu")
    outdir = "/iopsstor/scratch/cscs/mrohania/audio_samples/dpo_ab"; os.makedirs(outdir, exist_ok=True)
    state = InferenceState(ci, mimi, spm, lm, 1, 1.0, dev, **ci.lm_gen_config)

    def decode(tt):
        ids = [int(t) for t in tt if int(t) >= 0 and int(t) not in (pad_id, spm.eos_id())]
        try: return spm.decode(ids).strip() if ids else ""
        except Exception: return ""
    def gen(scaling, user_pcm):
        set_scaling(lm, scaling)
        state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
        inp = np.concatenate([user_pcm, np.zeros(int(6.0*sr), np.float32)])
        with torch.no_grad(): out = state.run(torch.from_numpy(inp[None, None]).to(dev))
        return out[0][1][0].detach().cpu().numpy().astype(np.float32), decode(out[0][0])

    scenarios = [("cold", np.zeros(int(0.5*sr), np.float32))]
    for u in pick_user_turns(mimi, 2): scenarios.append(("resp", u))
    print(f"\n===== DPO A/B  {a.label} =====")
    for i, (kind, u) in enumerate(scenarios):
        ab, tb = gen(0.0, u)       # BEFORE (fine-tune)
        aa, ta = gen(None, u)      # AFTER (DPO)
        sphn.write_wav(f"{outdir}/{a.label}_{i}_{kind}_BEFORE.wav", ab[None], sr)
        sphn.write_wav(f"{outdir}/{a.label}_{i}_{kind}_AFTER.wav", aa[None], sr)
        print(f"[{kind} {i}] BEFORE: {tb[:100]!r}")
        print(f"[{kind} {i}] AFTER : {ta[:100]!r}")
    print(f"saved to {outdir}/{a.label}_*", flush=True)

if __name__ == "__main__":
    main()
