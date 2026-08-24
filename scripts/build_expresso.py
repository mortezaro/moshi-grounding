#!/usr/bin/env python
"""Expresso (real, clean 48kHz, expressive) -> conditioned stereo for Moshi.
Moshi channel = concatenated same-style expressive utterances (resampled 24k); user channel silent.
text_conditions={emotion} from Expresso style. Clean acoustics + real emotion prosody."""
import os, re, json, glob, argparse, collections, tempfile
import numpy as np, sphn
import pyarrow.parquet as pq
SR=24000
EMAP={"happy":"hap","sad":"sad","default":"neu","enunciated":"neu"}
NPER=15      # utterances per clip
GAP=0.3      # seconds between utterances
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default="/iopsstor/scratch/cscs/mrohania/datasets/DailyExpressoCond")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    a=ap.parse_args()
    ow=os.path.join(a.out,"data_stereo"); os.makedirs(ow,exist_ok=True)
    fs=sorted(glob.glob("/iopsstor/scratch/cscs/mrohania/hf_cache/datasets--ylacombe--expresso/snapshots/*/**/*.parquet", recursive=True))
    rows=[]
    for f in fs:
        for r in pq.read_table(f, columns=["audio","text","speaker_id","style"]).to_pylist():
            if r["style"] in EMAP: rows.append(r)
    grp=collections.defaultdict(list)
    for r in rows: grp[(r["speaker_id"], r["style"])].append(r)
    keys=sorted(grp.keys()); keys=[k for i,k in enumerate(keys) if i%a.nshards==a.shard]
    def dec(b):
        with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as tf: tf.write(b); tp=tf.name
        try: d,sr=sphn.read(tp)
        finally: os.unlink(tp)
        w=d.mean(0) if d.ndim>1 else d
        if sr!=SR: w=sphn.resample(w,sr,SR)
        return np.asarray(w,dtype=np.float32)
    from collections import Counter; st=Counter(); n=0
    jo=open(os.path.join(a.out,f"shard_{a.shard}.jsonl"),"w")
    for k in keys:
        utts=grp[k]; emo=EMAP[k[1]]
        for c in range(0, len(utts), NPER):
            chunk=utts[c:c+NPER]
            if len(chunk)<4: continue
            segs=[]; aligns=[]; t=0.0
            for r in chunk:
                try: w=dec(r["audio"]["bytes"])
                except Exception: continue
                dur=len(w)/SR
                txt=re.sub(r"<[^>]+>","",r["text"] or "").split()
                if txt and dur>0:
                    dt=dur/len(txt)
                    for i,wd in enumerate(txt): aligns.append([wd,[round(t+i*dt,3),round(t+(i+1)*dt,3)],"SPEAKER_MAIN"])
                segs.append(w); segs.append(np.zeros(int(GAP*SR),dtype=np.float32)); t+=dur+GAP
            if not aligns: continue
            moshi=np.concatenate(segs); total=len(moshi)
            buf=np.zeros((2,total),dtype=np.float32); buf[0]=moshi   # ch1 (user) silent
            pk=np.abs(buf).max();
            if pk>0: buf=buf*(0.95/pk)
            bn=f"{a.shard}_{n}"
            sphn.write_wav(os.path.join(ow,bn+".wav"),buf,SR)
            json.dump({"alignments":aligns,"text_conditions":{"emotion":emo}},open(os.path.join(ow,bn+".json"),"w"))
            jo.write(json.dumps({"path":f"data_stereo/{bn}.wav","duration":total/SR})+"\n"); st[emo]+=1; n+=1
    jo.close(); print(f"[{a.shard}] DONE clips={n} emo={dict(st)}")
if __name__=="__main__": main()
