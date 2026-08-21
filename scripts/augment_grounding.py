#!/usr/bin/env python
"""Unified full-grounding augmentation: injects emotion + cognitive + environment
token families into moshi-format stereo conversation data. keep_and_shift-safe.
Works on any dataset with stereo wav + sibling word-alignment .json + jsonl.
Families (--families): emotion,latency,env,cognitive  (comma list; default all).
Signal degeneracy note: env/overlap/proximity are ~constant on clean studio audio
(DailyTalk) but carry real variance on naturalistic corpora (CANDOR/Fisher)."""
import os, json, glob, argparse, numpy as np, torch, sphn, ser_dim, signals as S, labeler
SR=16000; R_THR=0.02; MIN_GAP=0.30; EPS=0.08
_TH=json.load(open(os.path.join(os.path.dirname(__file__),"thresholds.json")))
LAT_E=[0.45,0.70]; RATE_E=[2.5,4.5]; DISF_E=[0.08]; PROX_E=[-19.0,-15.0]; SNR_E=[15.0,45.0]

def emo(a,vl):
    ah=a>=_TH["aro_hi"]; al=a<=_TH["aro_lo"]; vh=vl>=_TH["val_hi"]; vll=vl<=_TH["val_lo"]
    if ah and vll: return "ang"
    if ah and vh: return "hap"
    if al and vll: return "sad"
    if vh: return "hap"
    if vll: return "sad"
    return "neu"
def rms(x): return float(np.sqrt((x.astype(np.float32)**2).mean()+1e-9))
def main_turns(al):
    turns=[]; cur=None
    for w,ts,s in al:
        if cur is None: cur=[ts[0],ts[1],[w]]
        elif ts[0]-cur[1]<=0.6: cur[1]=ts[1]; cur[2].append(w)
        else: turns.append(tuple(cur)); cur=[ts[0],ts[1],[w]]
    if cur: turns.append(tuple(cur))
    return turns

def process(jf, wf, handle, dev, fams):
    al=json.load(open(jf))["alignments"]
    if not al: return None
    turns=main_turns(al)
    d,_=sphn.read(wf, sample_rate=SR); L,R=d[0],d[1]
    ovl=S.overlap_fraction(L,R,SR)
    # batch SER over self (L) turns + user (R) gaps
    specs=[]; meta=[]; prev=0.0
    for ti,(t0,t1,words) in enumerate(turns):
        g0,g1=prev,t0
        if g1-g0>=MIN_GAP:
            gseg=R[int(g0*SR):int(g1*SR)]
            if rms(gseg)>R_THR:
                specs.append(gseg); meta.append(("user",ti,g0,g1))
        specs.append(L[int(t0*SR):int(t1*SR)]); meta.append(("self",ti,t0,t1))
        prev=t1
    emos=labeler.predict(handle,dev,specs) if specs else []
    turn_tags={}  # ti -> {"user":[tags], "self":[tags]}
    for (kind,ti,s0,s1),em in zip(meta,emos):
        tags=[]
        if kind=="user":
            gseg=R[int(s0*SR):int(s1*SR)]
            if "emotion" in fams: tags.append(f"[user {em}]")
            if "latency" in fams:
                lat=S.response_latency(gseg,SR,s0,s0)
                tags.append(f"[lat {S.bucketize(lat,LAT_E,['fast','med','slow'])}]")
            if "env" in fams:
                tags.append(f"[env {S.bucketize(S.snr_db(gseg,SR),SNR_E,['noisy','mild','clean'])}]")
                tags.append(f"[prox {S.bucketize(S.proximity_level(gseg,SR),PROX_E,['far','mid','near'])}]")
                tags.append(f"[ovl {'y' if ovl>0.05 else 'n'}]")
        else:  # self
            words=turns[ti][2]
            if "emotion" in fams: tags.append(f"[self {em}]")
            if "cognitive" in fams:
                tags.append(f"[rate {S.bucketize(S.speech_rate(words,s0,s1),RATE_E,['slow','norm','fast'])}]")
                tags.append(f"[disf {S.bucketize(S.disfluency_rate(words),DISF_E,['lo','hi'])}]")
        turn_tags.setdefault(ti,{})[kind]=tags
    # inject: pack all tags into the pre-turn gap, chronological, positive-duration
    inj=[]
    for ti,(t0,t1,_) in enumerate(turns):
        blk=turn_tags.get(ti,{})
        seq=blk.get("user",[])+blk.get("self",[])   # perception then expression
        end=t0-0.02
        for tag in reversed(seq):
            s=end-EPS
            if s<=0: break
            inj.append([tag,[round(s,3),round(end,3)],"SPEAKER_MAIN"])
            end=s-0.005
    merged=[a for a in (al+inj) if a[1][1]>a[1][0]]
    merged.sort(key=lambda a:a[1][0])
    return {"alignments":merged}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", default="/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkContiguous")
    ap.add_argument("--out", default="/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkGround")
    ap.add_argument("--families", default="emotion,latency,env,cognitive")
    ap.add_argument("--labeler", default="audeering")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a=ap.parse_args()
    fams=set(a.families.split(","))
    dev="cuda" if torch.cuda.is_available() else "cpu"
    handle=labeler.load(a.labeler, dev)
    sw=os.path.join(a.src,"data_stereo"); ow=os.path.join(a.out,"data_stereo"); os.makedirs(ow,exist_ok=True)
    files=sorted(glob.glob(sw+"/*.wav"), key=lambda p:int(os.path.basename(p)[:-4]) if os.path.basename(p)[:-4].isdigit() else 0)
    if a.limit: files=files[:a.limit]
    mine=[f for i,f in enumerate(files) if i%a.nshards==a.shard]
    from collections import Counter; st=Counter(); n=0
    with open(os.path.join(a.out,f"shard_{a.shard}.jsonl"),"w") as jo:
        for wf in mine:
            jf=wf[:-4]+".json"
            if not os.path.exists(jf): continue
            try: res=process(jf,wf,handle,dev,fams)
            except Exception as ex: print("ERR",os.path.basename(wf),ex); continue
            if not res: continue
            bn=os.path.basename(wf); link=os.path.join(ow,bn)
            if not os.path.lexists(link): os.symlink(wf,link)
            json.dump(res, open(os.path.join(ow,bn[:-4]+".json"),"w"))
            d,_=sphn.read(wf,sample_rate=SR)
            jo.write(json.dumps({"path":f"data_stereo/{bn}","duration":d.shape[1]/SR})+"\n")
            for a2 in res["alignments"]:
                if a2[0].startswith("["): st[a2[0].split()[0][1:]]+=1
            n+=1
            if n%100==0: print(f"[shard {a.shard}] {n}/{len(mine)}",flush=True)
    print(f"[shard {a.shard}] DONE {n}. tag-family counts:", dict(st))
if __name__=="__main__": main()
