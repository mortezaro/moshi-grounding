#!/usr/bin/env python
"""Auto-label emotions and inject [self X]/[user X] tokens into DailyTalk transcripts.
Builds a mirror dataset (wavs symlinked, augmented .json siblings, per-shard jsonl).
Shardable across GPUs: --shard i --nshards N."""
import os, json, glob, argparse, numpy as np, torch, sphn
import json as _json, ser_dim, signals as _sig

SR=16000
_LAT_EDGES=[0.45,0.70]; _LAT_LABELS=["fast","med","slow"]
_TH=_json.load(open(os.path.join(os.path.dirname(__file__),"thresholds.json")))
def _emo(a,vl):
    ah=a>=_TH["aro_hi"]; al_=a<=_TH["aro_lo"]; vh=vl>=_TH["val_hi"]; vl_=vl<=_TH["val_lo"]
    if ah and vl_: return "ang"
    if ah and vh: return "hap"
    if al_ and vl_: return "sad"
    if vh: return "hap"
    if vl_: return "sad"
    return "neu"
R_ENERGY_THR=0.02       # right-channel RMS to count a gap as a real user turn
MIN_USER_GAP=0.30       # seconds
EPS=0.08                # width of an injected tag in seconds

def rms(x): return float(np.sqrt((x.astype(np.float32)**2).mean()+1e-9))

def main_turns(al):
    """Group consecutive SPEAKER_MAIN words into (t0,t1,wordcount)."""
    turns=[]; cur=None
    for w,ts,s in al:
        if cur is None: cur=[ts[0],ts[1],1]
        elif ts[0]-cur[1] <= 0.6:  # same turn if gap<0.6s
            cur[1]=ts[1]; cur[2]+=1
        else: turns.append(tuple(cur)); cur=[ts[0],ts[1],1]
    if cur: turns.append(tuple(cur))
    return turns

def label_batch(proc, model, dev, segs):
    if not segs: return []
    vad=ser_dim.predict(proc, model, dev, segs)   # [aro,dom,val]
    return [_emo(a, vl) for a,_,vl in vad]

def process_file(jf, wf, fe, model, dev):
    al=json.load(open(jf))["alignments"]
    if not al: return None
    turns=main_turns(al)
    d,_=sphn.read(wf, sample_rate=SR)          # [2, N]
    L,R=d[0],d[1]; dur=L.shape[0]/SR
    # collect segments to label: each MAIN turn (left ch) + each preceding gap (right ch)
    seg_specs=[]; _lat_by_turn={}  # (kind, turn_idx, seg_audio)
    prev_end=0.0
    for ti,(t0,t1,_) in enumerate(turns):
        gap0,gap1=prev_end,t0
        if gap1-gap0>=MIN_USER_GAP:
            gseg=R[int(gap0*SR):int(gap1*SR)]
            if rms(gseg)>R_ENERGY_THR:
                seg_specs.append(("user",ti,gseg))
                lat=_sig.response_latency(gseg,SR,gap0,gap0)
                _lat_by_turn[ti]=_sig.bucketize(lat,_LAT_EDGES,_LAT_LABELS)
        seg_specs.append(("self",ti,L[int(t0*SR):int(t1*SR)]))
        prev_end=t1
    labels=label_batch(fe,model,dev,[s[2] for s in seg_specs])
    # map turn_idx -> {self:emo, user:emo}
    emo={}
    for (kind,ti,_),lab in zip(seg_specs,labels):
        emo.setdefault(ti,{})[kind]=lab
    # build injected synthetic alignments
    inj=[]
    for ti,(t0,t1,_) in enumerate(turns):
        e=emo.get(ti,{})
        slot_end=t0-0.02
        if "self" in e:
            s1=slot_end-EPS
            inj.append([f"[self {e['self']}]",[round(max(s1,0.0),3),round(slot_end,3)],"SPEAKER_MAIN"])
            slot_end=s1-0.01
        if "user" in e:
            u1=slot_end-EPS
            if u1>0:
                inj.append([f"[user {e['user']}]",[round(u1,3),round(slot_end,3)],"SPEAKER_MAIN"])
                slot_end=u1-0.01
                if ti in _lat_by_turn and slot_end-EPS>0:
                    l1=slot_end-EPS
                    inj.append([f"[lat {_lat_by_turn[ti]}]",[round(l1,3),round(slot_end,3)],"SPEAKER_MAIN"])
    # merge injected + original, sort by start time, drop non-positive durations
    merged=[a for a in (al+inj) if a[1][1]>a[1][0]]
    merged.sort(key=lambda a:a[1][0])
    return {"alignments":merged}, emo

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", default="/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkContiguous")
    ap.add_argument("--out", default="/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkState")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    fe,model=ser_dim.load(dev)
    src_wav=os.path.join(a.src,"data_stereo")
    out_wav=os.path.join(a.out,"data_stereo"); os.makedirs(out_wav,exist_ok=True)
    files=sorted(glob.glob(src_wav+"/*.wav"), key=lambda p:int(os.path.basename(p)[:-4]) if os.path.basename(p)[:-4].isdigit() else 0)
    if a.limit: files=files[:a.limit]
    mine=[f for i,f in enumerate(files) if i%a.nshards==a.shard]
    jsonl=os.path.join(a.out,f"shard_{a.shard}.jsonl")
    from collections import Counter; stats=Counter()
    n=0
    with open(jsonl,"w") as jf_out:
        for wf in mine:
            jf=wf[:-4]+".json"
            if not os.path.exists(jf): continue
            try:
                res=process_file(jf,wf,fe,model,dev)
            except Exception as ex:
                print("ERR",wf,ex); continue
            if res is None: continue
            aug,emo=res
            bn=os.path.basename(wf)
            # symlink wav, write augmented json
            link=os.path.join(out_wav,bn)
            if not os.path.lexists(link): os.symlink(wf,link)
            json.dump(aug, open(os.path.join(out_wav,bn[:-4]+".json"),"w"))
            d,_=sphn.read(wf, sample_rate=SR)
            jf_out.write(json.dumps({"path":f"data_stereo/{bn}","duration":d.shape[1]/SR})+"\n")
            for ti,e in emo.items():
                for k,v in e.items(): stats[f"{k}:{v}"]+=1
            n+=1
            if n%100==0: print(f"[shard {a.shard}] {n}/{len(mine)}", flush=True)
    print(f"[shard {a.shard}] DONE {n} files. label stats:", dict(stats))

if __name__=="__main__": main()
