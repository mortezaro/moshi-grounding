#!/usr/bin/env python
"""Systematic per-stage evaluator. Teacher-forced probe over ALL grounding families.
Reports raw acc, BALANCED acc (macro-recall, robust to class imbalance), and the
majority-class baseline per family. Appends one JSON row to a ledger."""
import sys; sys.path.insert(0,"/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import argparse, os, glob, json, time, numpy as np, torch, sphn
from collections import Counter, defaultdict
from moshi.models import loaders
from finetune.data.interleaver import Interleaver, InterleavedTokenizer

FAMILIES={
 "user":["neu","hap","ang","sad"], "self":["neu","hap","ang","sad"],
 "lat":["fast","med","slow"], "env":["clean","mild","noisy"],
 "prox":["near","mid","far"], "ovl":["y","n"],
 "rate":["slow","norm","fast"], "disf":["lo","hi"],
 "aro":["lo","mid","hi"], "dom":["lo","mid","hi"], "pit":["lo","mid","hi"],
}
def fam_token_maps(spm):
    maps={}
    for fam,vals in FAMILIES.items():
        ctx=None; vmap={}
        for v in vals:
            ids=spm.encode("["+fam+" "+v+"]")
            ctx=ids[1]; vmap[v]=ids[-2]
        maps[fam]=(ctx,vmap)
    return maps

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--adapter", default=""); ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-bf16")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=80); ap.add_argument("--duration", type=float, default=100.0)
    ap.add_argument("--stage", required=True); ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--ledger", default="/iopsstor/scratch/cscs/mrohania/moshi_exp/ledger.jsonl")
    a=ap.parse_args(); dev="cuda"
    if a.baseline or not a.adapter:
        ci=loaders.CheckpointInfo.from_hf_repo(a.hf_repo); lm=ci.get_moshi(device=dev,dtype=torch.bfloat16)
    else:
        cfg=os.path.join(os.path.dirname(a.adapter),"config.json")
        ci=loaders.CheckpointInfo.from_hf_repo(a.hf_repo,lora_weights=a.adapter,config_path=cfg)
        lm=ci.get_moshi(device=dev,dtype=torch.bfloat16,fuse_lora=True)
    mimi=ci.get_mimi(device=dev); spm=ci.get_text_tokenizer()
    itl=Interleaver(spm,mimi.frame_rate,lm.text_padding_token_id,lm.end_of_text_padding_id,lm.zero_token_id,keep_main_only=True,keep_and_shift=True)
    itok=InterleavedTokenizer(mimi,itl,duration_sec=a.duration)
    fm=fam_token_maps(spm)
    valid={}; ctxof={}
    for fam,(ctx,vmap) in fm.items():
        ctxof[fam]=ctx
        for v,tid in vmap.items(): valid.setdefault(tid,[]).append((fam,v))
    conf=defaultdict(Counter)
    files=sorted(glob.glob(a.data+"/data_stereo/*.wav"))[-a.n:]
    for wf in files:
        try:
            wav,_=sphn.read(wf,sample_rate=mimi.sample_rate)
            with torch.no_grad():
                codes=itok(wav,0.0,wf).codes.to(dev); out=lm(codes=codes)
            tl=out.text_logits[0,0].float().cpu(); gt=codes[0,0].cpu().tolist()
        except Exception as ex:
            print("skip",os.path.basename(wf),ex); continue
        for t in range(2,len(gt)):
            cands=valid.get(gt[t])
            if not cands: continue
            win=gt[max(0,t-3):t]; fam=None; true=None
            for f,v in cands:
                if ctxof[f] in win: fam=f; true=v; break
            if fam is None: continue
            ctx,vmap=fm[fam]; ids=list(vmap.values())
            sub=torch.tensor([tl[t][i] for i in ids]); pred=list(vmap.keys())[int(sub.argmax())]
            conf[fam][(true,pred)]+=1
    row={"stage":a.stage,"step":a.step,"dataset":os.path.basename(a.data),
         "adapter":os.path.basename(os.path.dirname(os.path.dirname(a.adapter))) if a.adapter else "baseline",
         "time":time.strftime("%Y-%m-%d %H:%M"),"n_files":len(files),
         "acc":{},"bal_acc":{},"maj":{},"n":{}}
    for f in FAMILIES:
        c=conf[f]; tot=sum(c.values()); row["n"][f]=tot
        if not tot:
            row["acc"][f]=row["bal_acc"][f]=row["maj"][f]=None; continue
        hit=sum(v for (tr,pr),v in c.items() if tr==pr)
        tc=Counter()
        for (tr,pr),v in c.items(): tc[tr]+=v
        recs=[c.get((cl,cl),0)/tc[cl] for cl in tc]
        row["acc"][f]=round(hit/tot,3)
        row["bal_acc"][f]=round(sum(recs)/len(recs),3)
        row["maj"][f]=round(max(tc.values())/tot,3)
    os.makedirs(os.path.dirname(a.ledger),exist_ok=True)
    open(a.ledger,"a").write(json.dumps(row)+"\n")
    print("STAGE",a.stage,"step",a.step,"dataset",row["dataset"])
    for f in FAMILIES:
        print("  %-6s acc=%s bal_acc=%s maj_base=%s (n=%s)"%(f,row["acc"][f],row["bal_acc"][f],row["maj"][f],row["n"][f]))
if __name__=="__main__": main()
