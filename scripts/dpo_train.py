#!/usr/bin/env python
"""Single-GPU DPO for a Moshi fine-tune. Efficient reference trick: fuse the base adapter
(e.g. E_long) into the weights, add ONE fresh trainable LoRA; the reference policy is the
same model with the new LoRA scaling set to 0 (no second model, no FSDP).

DPO loss (per pair):  -log sigmoid( beta * [ (lp_pol_c - lp_ref_c) - (lp_pol_r - lp_ref_r) ] )
where lp = sum of token log-probs over the response mask (text, or text+audio).

Settings (the sweep knobs): --beta --lr --steps --stream {text,both} --rank.
Evals before/after in-job (generate with LoRA off vs on) and prints conv-quality delta.
"""
import sys, os, glob, json, re, argparse, random
FT = "/iopsstor/scratch/cscs/mrohania/moshi-finetune"; sys.path.insert(0, FT)
import numpy as np, torch, sphn
import torch.nn.functional as F
from moshi.models import loaders
from moshi.models.lm import LMGen
from moshi.modules.lora import replace_all_linear_with_lora, LoRALinear
from moshi.run_inference import get_condition_tensors
from finetune.data.interleaver import Interleaver, InterleavedTokenizer

WH = re.compile(r"\b(what|why|how|when|where|who|which|do you|are you|would you|tell me|what about you)\b", re.I)

# ---------- lora scaling toggle (policy vs reference) ----------
def set_scaling(model, val):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if not hasattr(m, "_base_scaling"): m._base_scaling = m.scaling
            m.scaling = m._base_scaling if val is None else val

# ---------- sequence log-prob over the response mask ----------
def masked_logprob(logits, target, mask):
    logp = F.log_softmax(logits.float(), dim=-1)
    g = logp.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    g = torch.where(mask.bool(), g, torch.zeros_like(g))
    return g.sum()

def seq_logprob(model, codes, cond, audio_offset, dep_q, stream, ref):
    if ref: set_scaling(model, 0.0)
    try:
        cm = torch.no_grad() if ref else torch.enable_grad()
        with cm:
            out = model(codes=codes, condition_tensors=cond)
            lp = masked_logprob(out.text_logits, codes[:, :audio_offset], out.text_mask)
            if stream == "both":
                lp = lp + masked_logprob(out.logits, codes[:, audio_offset:audio_offset+dep_q], out.mask)
    finally:
        if ref: set_scaling(model, None)
    return lp

# ---------- lightweight conv scoring for in-job eval ----------
def score_text(txt):
    words = re.findall(r"[a-zA-Z']+", txt.lower()); n = len(words)
    if n == 0: return 0.0
    bg = list(zip(words, words[1:])); rep = 1.0-(len(set(bg))/len(bg)) if bg else 0.0
    diversity = len(set(words))/n
    asks = 1.0 if ("?" in txt or WH.search(txt)) else 0.0
    length_ok = 1.0 if 3 <= n <= 40 else (0.3 if n > 40 else 0.5)
    return float(0.4*(1-rep) + 0.25*diversity + 0.2*length_ok + 0.15*asks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--prefs", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--stream", default="text", choices=["text", "both"])
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--out", default="");
    a = ap.parse_args(); dev = "cuda"; torch.manual_seed(0)
    ci = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",
                                             lora_weights=a.adapter, config_path=a.config)
    lm = ci.get_moshi(device=dev, dtype=torch.bfloat16, fuse_lora=True)   # fuse base adapter
    mimi = ci.get_mimi(device=dev); mimi.eval()
    spm = ci.get_text_tokenizer(); sr = mimi.sample_rate
    audio_offset = lm.audio_offset; dep_q = lm.dep_q
    # add fresh trainable LoRA on top of the fused model
    replace_all_linear_with_lora(lm, a.rank, scaling=1.0, device=dev, dtype=torch.bfloat16)
    for m in lm.modules():                      # standard LoRA: zero-init B so initial delta = 0
        if isinstance(m, LoRALinear): torch.nn.init.zeros_(m.lora_B.weight)
    train_params = []
    for name, p in lm.named_parameters():
        p.requires_grad = ("lora_A" in name or "lora_B" in name)
        if p.requires_grad: train_params.append(p)
    print(f"trainable lora params: {sum(p.numel() for p in train_params)/1e6:.1f}M", flush=True)
    os.environ.setdefault("COND_EMOTION", "neu")
    with torch.no_grad():                        # constant condition (not training conditioner) -> no graph reuse
        cond = get_condition_tensors("moshi", lm, 1, 1.0)
    def _detach(c):
        if isinstance(c, dict): return {k: _detach(v) for k, v in c.items()}
        return c.detach() if torch.is_tensor(c) else c
    cond = _detach(cond)

    interleaver = Interleaver(spm, mimi.frame_rate, lm.text_padding_token_id,
                              lm.end_of_text_padding_id, lm.zero_token_id,
                              keep_main_only=True, keep_and_shift=True)
    tok = InterleavedTokenizer(mimi, interleaver, duration_sec=30.0)
    root = os.path.dirname(a.prefs.rstrip("/")) if a.prefs.endswith(".jsonl") else a.prefs
    lines = []
    for jf in glob.glob(os.path.join(a.prefs, "prefs_*.jsonl")):
        for ln in open(jf): lines.append((os.path.dirname(jf), json.loads(ln)))
    random.Random(0).shuffle(lines)
    print(f"loaded {len(lines)} preference pairs", flush=True)

    def codes_of(base, rel):
        wav, _ = sphn.read(os.path.join(base, rel), sample_rate=sr)
        s = tok(torch.from_numpy(wav), 0.0, os.path.join(base, rel))
        return s.codes.to(dev)

    # ---------- in-job eval: generate with lora off (ref) vs on (policy) ----------
    from moshi.run_inference import InferenceState
    def _decode(tt):
        ids = [int(t) for t in tt if int(t) >= 0 and int(t) not in (lm.text_padding_token_id, spm.eos_id())]
        try: return spm.decode(ids).strip() if ids else ""
        except Exception: return ""
    def evaluate():
        lm.eval(); set_scaling(lm, None)   # policy scaling (before training LoRA~=0 => baseline)
        state = InferenceState(ci, mimi, spm, lm, 1, 1.0, dev, **ci.lm_gen_config)
        scores = []
        for _ in range(6):
            state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
            inp = np.zeros(int(6.5*sr), np.float32)
            with torch.no_grad():
                out = state.run(torch.from_numpy(inp[None, None]).to(dev))
            scores.append(score_text(_decode(out[0][0])))
        lm._stop_streaming(); mimi._stop_streaming()   # exit streaming so training forward works
        return float(np.mean(scores))

    pre = evaluate(); print(f"[{a.label}] EVAL before-DPO conv={pre:.3f}", flush=True)

    opt = torch.optim.AdamW(train_params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    lm.train(); step = 0; acc = 0; running = 0.0; opt.zero_grad()
    while step < a.steps:
        for base, ex in lines:
            try:
                cc = codes_of(base, ex["chosen"]); cr = codes_of(base, ex["rejected"])
            except Exception as e:
                continue
            lp_pol_c = seq_logprob(lm, cc, cond, audio_offset, dep_q, a.stream, ref=False)
            lp_ref_c = seq_logprob(lm, cc, cond, audio_offset, dep_q, a.stream, ref=True)
            lp_pol_r = seq_logprob(lm, cr, cond, audio_offset, dep_q, a.stream, ref=False)
            lp_ref_r = seq_logprob(lm, cr, cond, audio_offset, dep_q, a.stream, ref=True)
            logits = a.beta * ((lp_pol_c - lp_ref_c) - (lp_pol_r - lp_ref_r))
            loss = -F.logsigmoid(logits) / a.accum
            loss.backward(); running += float(loss)*a.accum; acc += 1
            if acc % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 20 == 0:
                    print(f"[{a.label}] step {step}/{a.steps} dpo_loss={running/ (20*a.accum):.4f}", flush=True); running = 0.0
                if step >= a.steps: break
    post = evaluate()
    print(f"\n[{a.label}] ===== DPO RESULT beta={a.beta} lr={a.lr} stream={a.stream} rank={a.rank} =====")
    print(f"[{a.label}] conv  before={pre:.3f}  after={post:.3f}  delta={post-pre:+.3f}", flush=True)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        sd = {k: v.detach().cpu() for k, v in lm.state_dict().items() if "lora_A" in k or "lora_B" in k}
        from safetensors.torch import save_file
        save_file(sd, os.path.join(a.out, "dpo_lora.safetensors"))
        json.dump(dict(label=a.label, beta=a.beta, lr=a.lr, stream=a.stream, rank=a.rank,
                       conv_before=pre, conv_after=post),
                  open(os.path.join(a.out, "result.json"), "w"), indent=2)
        print(f"[{a.label}] saved new LoRA + result to {a.out}", flush=True)

if __name__ == "__main__":
    main()
