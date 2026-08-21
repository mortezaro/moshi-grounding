#!/usr/bin/env python
"""Compare emotion labelers on ser_english ground truth (4-class): SER4 (trained) vs
audeering-dimensional+calibration (current pipeline labeler)."""
import io, glob, wave, json, numpy as np, torch
import pyarrow.parquet as pq
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
import ser_dim
MAP={"neutral":"neu","happiness":"hap","anger":"ang","sadness":"sad"}
LAB=["neu","hap","ang","sad"]; L2I={l:i for i,l in enumerate(LAB)}
SR=16000
_TH=json.load(open("/iopsstor/scratch/cscs/mrohania/moshi_emo/thresholds.json"))
def emo(a,vl):
    ah=a>=_TH["aro_hi"]; al=a<=_TH["aro_lo"]; vh=vl>=_TH["val_hi"]; vll=vl<=_TH["val_lo"]
    if ah and vll: return "ang"
    if ah and vh: return "hap"
    if al and vll: return "sad"
    if vh: return "hap"
    if vll: return "sad"
    return "neu"
def decode(b):
    w=wave.open(io.BytesIO(b),"rb"); n=w.getnframes(); sr=w.getframerate(); ch=w.getnchannels()
    a=np.frombuffer(w.readframes(n),dtype=np.int16).astype(np.float32)/32768.0
    if ch>1: a=a.reshape(-1,ch).mean(1)
    return a,sr
dev="cuda"
# load held-out slice (last parquet)
X=[]; Y=[]
for pf in sorted(glob.glob("/iopsstor/scratch/cscs/mrohania/datasets/ser_english/data/*.parquet"))[1:4]:
    for r in pq.read_table(pf,columns=["audio","emotion"]).to_pylist():
        e=MAP.get(str(r["emotion"]).lower())
        if e is None: continue
        try:
            a,sr=decode(r["audio"]["bytes"])
            if sr!=SR or len(a)<0.3*SR or len(a)>12*SR: continue
            X.append(a); Y.append(e)
        except Exception: continue
        if len(X)>=900: break
print("held-out samples:", len(X), "dist:", {l:Y.count(l) for l in LAB})
# SER4
fe=AutoFeatureExtractor.from_pretrained("/iopsstor/scratch/cscs/mrohania/models/ser4")
m4=Wav2Vec2ForSequenceClassification.from_pretrained("/iopsstor/scratch/cscs/mrohania/models/ser4").to(dev).eval()
# audeering
proc,adm=ser_dim.load(dev)
def macc(preds):
    from collections import Counter
    tc=Counter(); hc=Counter()
    for p,y in zip(preds,Y):
        tc[y]+=1; hc[y]+= int(p==y)
    recs=[hc[c]/tc[c] for c in tc]; raw=sum(hc.values())/len(Y)
    return raw, sum(recs)/len(recs)
p4=[]; pa=[]; B=16
for i in range(0,len(X),B):
    xb=X[i:i+B]
    inp=fe(xb,sampling_rate=SR,return_tensors="pt",padding=True).input_values.to(dev)
    with torch.no_grad(): p4+=[LAB[j] for j in m4(inp).logits.argmax(-1).cpu().tolist()]
    vad=ser_dim.predict(proc,adm,dev,xb); pa+=[emo(a,vl) for a,_,vl in vad]
r4,b4=macc(p4); ra,ba=macc(pa)
print("SER4       raw_acc=%.3f  balanced_acc=%.3f"%(r4,b4))
print("audeering  raw_acc=%.3f  balanced_acc=%.3f"%(ra,ba))
open("/iopsstor/scratch/cscs/mrohania/moshi_exp/ser_compare.log","w").write("SER4 raw=%.3f bal=%.3f | audeering raw=%.3f bal=%.3f\n"%(r4,b4,ra,ba))
