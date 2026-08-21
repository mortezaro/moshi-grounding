#!/usr/bin/env python
"""Evaluate emotion-awareness of the fine-tuned Moshi.
Feeds held-out USER turns (right channel) to the model, decodes Moshi inner-monologue
text, extracts [user X]/[self Y] markers; reports recognition match vs the SER label."""
import argparse, json, glob, re, os, numpy as np, torch, sphn
from moshi.models import loaders
from moshi.run_inference import InferenceState

TAG_RE = re.compile(r"\[(user|self)\s+(neu|hap|ang|sad)\]")

def load_model(adapter, hf_repo, device, dtype):
    import os as _os
    cfg=_os.path.join(_os.path.dirname(adapter),"config.json")
    ci = loaders.CheckpointInfo.from_hf_repo(hf_repo, lora_weights=adapter, config_path=cfg)
    mimi = ci.get_mimi(device=device)
    tok  = ci.get_text_tokenizer()
    lm   = ci.get_moshi(device=device, dtype=dtype, fuse_lora=True)
    return ci, mimi, tok, lm

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-bf16")
    ap.add_argument("--src", default="/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkEmoDim")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    a=ap.parse_args()
    dtype={"bf16":torch.bfloat16,"fp16":torch.float16,"fp32":torch.float32}[a.dtype]
    ci,mimi,tok,lm=load_model(a.adapter,a.hf_repo,a.device,dtype)
    sr=mimi.sample_rate
    state=InferenceState(ci, mimi, tok, lm, 1, 1.0, a.device, **ci.lm_gen_config)
    # pick user turns from augmented data: [user X] tag positions tell us the gap; label = the tag
    files=sorted(glob.glob(a.src+"/data_stereo/*.wav"),
                 key=lambda p:int(os.path.basename(p)[:-4]) if os.path.basename(p)[:-4].isdigit() else 0)[-60:]
    hits=0; tot=0; emitted_self=0
    for wf in files:
        al=json.load(open(wf[:-4]+".json"))["alignments"]
        # find a [user X] tag and the following user gap audio = right channel just before next real word
        for i,(w,ts,s) in enumerate(al):
            m=TAG_RE.match(w)
            if not m or m.group(1)!="user": continue
            ser=m.group(2)
            # the user turn audio precedes this tag: use [tag_start-2.0, tag_start] on right channel
            t1=ts[0]; t0=max(0.0,t1-3.0)
            d,_=sphn.read(wf, start_sec=t0, duration_sec=t1-t0, sample_rate=sr)
            user_pcm=torch.from_numpy(d[1:2][None]).to(a.device)  # [1,1,T] right channel
            try:
                state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
                out=state.run(user_pcm)
            except Exception as ex:
                print("run err", os.path.basename(wf), ex); break
            # decode emitted text tokens
            text_toks=out[0][0].flatten().tolist() if out else []
            toks=[t for t in text_toks if t not in (0,3)]
            text="".join(tok.id_to_piece(t).replace("▁"," ") for t in toks)
            preds=TAG_RE.findall(text)
            pu=[e for k,e in preds if k=="user"]; psf=[e for k,e in preds if k=="self"]
            ok = pu and pu[0]==ser
            hits+= int(bool(ok)); tot+=1; emitted_self+= int(bool(psf))
            print(f"{os.path.basename(wf):10s} SER_user={ser} moshi_user={pu[:2]} moshi_self={psf[:2]} text[:60]={text[:60]!r}")
            break
        if tot>=a.n: break
    print(f"\nRECOGNITION match: {hits}/{tot}  |  turns where Moshi emitted a [self ] tag: {emitted_self}/{tot}")

if __name__=="__main__": main()
