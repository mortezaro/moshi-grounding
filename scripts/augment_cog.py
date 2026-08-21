#!/usr/bin/env python
"""LIGHT TEST: cognitive/physical grounding. Adds arousal + dominance (free, from the
audeering VAD we already run) + pitch (cheap DSP) as new token families, alongside
emotion+latency. Tests whether Moshi can learn these axes. keep_and_shift-safe."""
import os, json, glob, argparse, numpy as np, torch, sphn, ser_dim, signals as S
SR=16000; R_THR=0.02; MIN_GAP=0.30; EPS=0.08
_T=json.load(open(os.path.join(os.path.dirname(__file__),"thresholds.json")))
_T2=json.load(open(os.path.join(os.path.dirname(__file__),"thresholds2.json")))
def emo(a,vl):
    ah=a>=_T["aro_hi"]; al=a<=_T["aro_lo"]; vh=vl>=_T["val_hi"]; vll=vl<=_T["val_lo"]
    if ah and vll: return "ang"
    if ah and vh: return "hap"
    if al and vll: return "sad"
    if vh: return "hap"
    if vll: return "sad"
    return "neu"
def b3(v,lo,hi,labs=("lo","mid","hi")): return labs[0] if v<lo else (labs[2] if v>hi else labs[1])
def rms(x): return float(np.sqrt((x.astype(np.float32)**2).mean()+1e-9))
def main_turns(al):
    turns=[]; cur=None
    for w,ts,s in al:
        if cur is None: cur=[ts[0],ts[1]]
        elif ts[0]-cur[1]<=0.6: cur[1]=ts[1]
        else: turns.append(tuple(cur)); cur=[ts[0],ts[1]]
    if cur: turns.append(tuple(cur))
    return turns
def process(jf,wf,proc,model,dev):
    al=json.load(open(jf))["alignments"]
    if not al: return None
    turns=main_turns(al); d,_=sphn.read(wf,sample_rate=SR); L,R=d[0],d[1]
    specs=[]; meta=[]; prev=0.0
    for ti,(t0,t1) in enumerate(turns):
        g0,g1=prev,t0
        if g1-g0>=MIN_GAP:
            g=R[int(g0*SR):int(g1*SR)]
            if rms(g)>R_THR: specs.append(g); meta.append(("user",ti))
        specs.append(L[int(t0*SR):int(t1*SR)]); meta.append(("self",ti)); prev=t1
    vad=ser_dim.predict(proc,model,dev,specs) if specs else []
    tags={}
    for (kind,ti),(a,dm,vl),seg in zip(meta,vad,specs):
        pit=S.pitch_hz(seg,SR)
        blk=[f"[{kind} {emo(a,vl)}]",
             f"[aro {b3(a,_T['aro_lo'],_T['aro_hi'])}]",
             f"[dom {b3(dm,_T2['dom_lo'],_T2['dom_hi'])}]"]
        if pit>0: blk.append(f"[pit {b3(pit,_T2['pit_lo'],_T2['pit_hi'])}]")
        tags.setdefault(ti,{})[kind]=blk
    inj=[]
    for ti,(t0,t1) in enumerate(turns):
        seq=tags.get(ti,{}).get("user",[])+tags.get(ti,{}).get("self",[])
        end=t0-0.02
        for tag in reversed(seq):
            s=end-EPS
            if s<=0: break
            inj.append([tag,[round(s,3),round(end,3)],"SPEAKER_MAIN"]); end=s-0.005
    merged=[a for a in (al+inj) if a[1][1]>a[1][0]]; merged.sort(key=lambda a:a[1][0])
    return {"alignments":merged}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src",default="/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkContiguous")
    ap.add_argument("--out",default="/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkCog")
    ap.add_argument("--shard",type=int,default=0); ap.add_argument("--nshards",type=int,default=1)
    ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args(); dev="cuda"; proc,model=ser_dim.load(dev)
    sw=os.path.join(a.src,"data_stereo"); ow=os.path.join(a.out,"data_stereo"); os.makedirs(ow,exist_ok=True)
    files=sorted(glob.glob(sw+"/*.wav"),key=lambda p:int(os.path.basename(p)[:-4]) if os.path.basename(p)[:-4].isdigit() else 0)
    if a.limit: files=files[:a.limit]
    mine=[f for i,f in enumerate(files) if i%a.nshards==a.shard]
    from collections import Counter; st=Counter(); n=0
    with open(os.path.join(a.out,f"shard_{a.shard}.jsonl"),"w") as jo:
        for wf in mine:
            jf=wf[:-4]+".json"
            if not os.path.exists(jf): continue
            try: res=process(jf,wf,proc,model,dev)
            except Exception as ex: print("ERR",os.path.basename(wf),ex); continue
            if not res: continue
            bn=os.path.basename(wf); lk=os.path.join(ow,bn)
            if not os.path.lexists(lk): os.symlink(wf,lk)
            json.dump(res,open(os.path.join(ow,bn[:-4]+".json"),"w"))
            d,_=sphn.read(wf,sample_rate=SR); jo.write(json.dumps({"path":f"data_stereo/{bn}","duration":d.shape[1]/SR})+"\n")
            for x in res["alignments"]:
                if x[0].startswith("["): st[x[0].split()[0][1:]]+=1
            n+=1
            if n%100==0: print(f"[{a.shard}] {n}/{len(mine)}",flush=True)
    print(f"[{a.shard}] DONE {n}. families:",dict(st))
if __name__=="__main__": main()
