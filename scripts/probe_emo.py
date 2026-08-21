#!/usr/bin/env python
"""Teacher-forced probe: does the fine-tuned model predict the right emotion token
at [self X]/[user Y] marker positions, given audio + context? 4-way accuracy."""
import sys; sys.path.insert(0, "/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import argparse, os, glob, json, numpy as np, torch, sphn
from moshi.models import loaders
from finetune.data.interleaver import Interleaver, InterleavedTokenizer

EMO={14187:"neu",10986:"hap",2169:"ang",10861:"sad"}   # emotion-word SP ids
SELF_CTX=5198; USER_CTX=2518
EMO_IDS=list(EMO.keys())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-bf16")
    ap.add_argument("--src", default="/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkEmoDim")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--duration", type=float, default=100.0)
    ap.add_argument("--baseline", action="store_true", help="load base model without adapter")
    a=ap.parse_args()
    dev="cuda"
    cfg=os.path.join(os.path.dirname(a.adapter),"config.json")
    if a.baseline:
        ci=loaders.CheckpointInfo.from_hf_repo(a.hf_repo)
        lm=ci.get_moshi(device=dev, dtype=torch.bfloat16)
    else:
        ci=loaders.CheckpointInfo.from_hf_repo(a.hf_repo, lora_weights=a.adapter, config_path=cfg)
        lm=ci.get_moshi(device=dev, dtype=torch.bfloat16, fuse_lora=True)
    mimi=ci.get_mimi(device=dev); spm=ci.get_text_tokenizer()
    itl=Interleaver(spm, mimi.frame_rate, lm.text_padding_token_id, lm.end_of_text_padding_id,
                    lm.zero_token_id, keep_main_only=True, keep_and_shift=True)
    itok=InterleavedTokenizer(mimi, itl, duration_sec=a.duration)

    from collections import Counter, defaultdict
    conf=defaultdict(Counter); tot=defaultdict(int); hit=defaultdict(int); rankhit=defaultdict(int)
    files=sorted(glob.glob(a.src+"/data_stereo/*.wav"),
                 key=lambda p:int(os.path.basename(p)[:-4]) if os.path.basename(p)[:-4].isdigit() else 0)[-a.n:]
    for wf in files:
        try:
            wav,_=sphn.read(wf, sample_rate=mimi.sample_rate)
            with torch.no_grad():
                codes=itok(wav, 0.0, wf).codes.to(dev)
                out=lm(codes=codes)
            tl=out.text_logits[0,0].float().cpu()   # [T, card]
            gt=codes[0,0].cpu().tolist()             # [T]
        except Exception as ex:
            print("skip", os.path.basename(wf), ex); continue
        T=len(gt)
        for t in range(2,T):
            if gt[t] not in EMO: continue
            win=gt[max(0,t-3):t]
            kind = "self" if SELF_CTX in win else ("user" if USER_CTX in win else None)
            if kind is None: continue
            true=EMO[gt[t]]
            logit_t=tl[t]
            # 4-way restricted prediction
            sub=torch.tensor([logit_t[i] for i in EMO_IDS])
            pred=EMO[EMO_IDS[int(sub.argmax())]]
            # full-vocab rank of the true token (is it in top-5?)
            topk=set(torch.topk(logit_t, 5).indices.tolist())
            tot[kind]+=1; hit[kind]+= int(pred==true); rankhit[kind]+= int(gt[t] in topk)
            conf[kind][f"{true}->{pred}"]+=1
    tag="BASELINE(no adapter)" if a.baseline else "FINE-TUNED"
    print(f"\n===== {tag} teacher-forced probe =====")
    for kind in ["user","self"]:
        n=tot[kind] or 1
        print(f"[{kind}] n={tot[kind]:4d}  4-way acc={hit[kind]/n:.3f}  (chance .25)  true-in-top5={rankhit[kind]/n:.3f}")
    for kind in ["user","self"]:
        print(f"  {kind} confusion (top):", dict(conf[kind].most_common(8)))

if __name__=="__main__": main()
