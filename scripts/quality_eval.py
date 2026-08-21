#!/usr/bin/env python
"""STANDING QUALITY GATE: is the fine-tuned model still a good Moshi?
Measures cross-entropy on PLAIN speech (no grounding tags) for BOTH streams:
  - text CE  (inner-monologue / language modeling)
  - audio CE (the speech codebooks = actual voice quality; first codebook = semantic)
Compare each stage to base. Rule: grounding must NOT raise audio CE materially."""
import sys; sys.path.insert(0,"/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import argparse, os, glob, torch, sphn
from moshi.models import loaders
from finetune.data.interleaver import Interleaver, InterleavedTokenizer

def run(adapter, data, n, duration=100.0):
    dev="cuda"
    if not adapter:
        ci=loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16"); lm=ci.get_moshi(device=dev,dtype=torch.bfloat16)
    else:
        cfg=os.path.join(os.path.dirname(adapter),"config.json")
        ci=loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",lora_weights=adapter,config_path=cfg)
        lm=ci.get_moshi(device=dev,dtype=torch.bfloat16,fuse_lora=True)
    mimi=ci.get_mimi(device=dev); spm=ci.get_text_tokenizer()
    itl=Interleaver(spm,mimi.frame_rate,lm.text_padding_token_id,lm.end_of_text_padding_id,lm.zero_token_id,keep_main_only=True,keep_and_shift=True)
    itok=InterleavedTokenizer(mimi,itl,duration_sec=duration)
    ao=lm.audio_offset; dq=lm.dep_q
    tpad={lm.text_padding_token_id,lm.end_of_text_padding_id,lm.zero_token_id}
    from moshi.run_inference import get_condition_tensors
    ct = get_condition_tensors("moshi", lm, 1, 1.0) if getattr(lm,"fuser",None) is not None else None
    tsum=0.0;tc=0; asum=0.0;ac=0; a0sum=0.0;a0c=0
    for wf in sorted(glob.glob(data+"/data_stereo/*.wav"))[-n:]:
        try:
            wav,_=sphn.read(wf,sample_rate=mimi.sample_rate)
            with torch.no_grad():
                codes=itok(wav,0.0,wf).codes.to(dev); out=lm(codes=codes, condition_tensors=ct)
            # text CE
            tl=out.text_logits[0,0].float(); gt=codes[0,0]
            m=torch.ones_like(gt,dtype=torch.bool)
            for p in tpad: m&=(gt!=p)
            if m.sum()>0:
                tsum+=float(torch.nn.functional.cross_entropy(tl[m],gt[m])); tc+=1
            # audio CE (all codebooks + first codebook separately)
            alog=out.logits[0].float()             # [K,T,card]
            atgt=codes[0, ao:ao+dq]                # [K,T]
            am=out.mask[0] if out.mask is not None else torch.ones_like(atgt,dtype=torch.bool)
            K=atgt.shape[0]
            ces=[]
            for k in range(K):
                mk=am[k]
                if mk.sum()==0: continue
                ce=float(torch.nn.functional.cross_entropy(alog[k][mk], atgt[k][mk]))
                ces.append(ce)
                if k==0: a0sum+=ce; a0c+=1
            if ces: asum+=sum(ces)/len(ces); ac+=1
        except Exception as ex: print("skip",os.path.basename(wf),ex)
    return (tsum/max(tc,1), asum/max(ac,1), a0sum/max(a0c,1), tc)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--adapter", default=""); ap.add_argument("--label", required=True)
    ap.add_argument("--data", default="/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkContiguous")
    ap.add_argument("--n", type=int, default=40)
    a=ap.parse_args()
    t,au,a0,cnt=run(a.adapter,a.data,a.n)
    print("QUALITY %-16s text_CE=%.4f  audio_CE=%.4f  audio_cb0_CE=%.4f  (n=%d)"%(a.label,t,au,a0,cnt))
    open("/iopsstor/scratch/cscs/mrohania/moshi_exp/quality.log","a").write("%s\ttext=%.4f\taudio=%.4f\tcb0=%.4f\tn=%d\n"%(a.label,t,au,a0,cnt))
