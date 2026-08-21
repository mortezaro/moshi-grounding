#!/usr/bin/env python
"""Build CONDITIONED data: plain word alignments (fluent speech, NO inline tokens) +
per-clip text_conditions={emotion} from SER. Emotion rides as a conditioning INPUT."""
import os, json, glob, argparse, numpy as np, torch, sphn, ser_dim
SR=16000
_T=json.load(open("/iopsstor/scratch/cscs/mrohania/moshi_emo/thresholds.json"))
def emo(a,vl):
    ah=a>=_T["aro_hi"]; al=a<=_T["aro_lo"]; vh=vl>=_T["val_hi"]; vll=vl<=_T["val_lo"]
    if ah and vll: return "ang"
    if ah and vh: return "hap"
    if al and vll: return "sad"
    if vh: return "hap"
    if vll: return "sad"
    return "neu"
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", default="/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkContiguous")
    ap.add_argument("--out", default="/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkCond")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a=ap.parse_args(); dev="cuda"; proc,model=ser_dim.load(dev)
    sw=os.path.join(a.src,"data_stereo"); ow=os.path.join(a.out,"data_stereo"); os.makedirs(ow,exist_ok=True)
    files=sorted(glob.glob(sw+"/*.wav"), key=lambda p:int(os.path.basename(p)[:-4]) if os.path.basename(p)[:-4].isdigit() else 0)
    if a.limit: files=files[:a.limit]
    mine=[f for i,f in enumerate(files) if i%a.nshards==a.shard]
    from collections import Counter; st=Counter(); n=0
    with open(os.path.join(a.out,f"shard_{a.shard}.jsonl"),"w") as jo:
        for wf in mine:
            jf=wf[:-4]+".json"
            if not os.path.exists(jf): continue
            al=json.load(open(jf))["alignments"]
            if not al: continue
            d,_=sphn.read(wf,sample_rate=SR)
            # main-speaker turns (left channel) -> dominant emotion
            segs=[]; cur=None; turns=[]
            for w,ts,s in al:
                if cur is None: cur=[ts[0],ts[1]]
                elif ts[0]-cur[1]<=0.6: cur[1]=ts[1]
                else: turns.append(tuple(cur)); cur=[ts[0],ts[1]]
            if cur: turns.append(tuple(cur))
            for t0,t1 in turns: segs.append(d[0][int(t0*SR):int(t1*SR)])
            vad=ser_dim.predict(proc,model,dev,segs) if segs else []
            emos=[emo(x[0],x[2]) for x in vad]
            dom = Counter(emos).most_common(1)[0][0] if emos else "neu"
            st[dom]+=1
            bn=os.path.basename(wf); lk=os.path.join(ow,bn)
            if not os.path.lexists(lk): os.symlink(wf,lk)
            json.dump({"alignments":al, "text_conditions":{"emotion":dom}}, open(os.path.join(ow,bn[:-4]+".json"),"w"))
            jo.write(json.dumps({"path":f"data_stereo/{bn}","duration":d.shape[1]/SR})+"\n"); n+=1
            if n%100==0: print(f"[{a.shard}] {n}",flush=True)
    print(f"[{a.shard}] DONE {n}. emotion conditions:",dict(st))
if __name__=="__main__": main()
