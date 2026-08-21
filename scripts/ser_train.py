#!/usr/bin/env python
"""Item 4: train a real 4-class speech-emotion classifier on ground-truth ser_english
(IEMOCAP+). Upgrades the calibrated-but-approximate audeering labeler for the next stage.
Reads parquet directly (decode audio bytes; avoids torchcodec)."""
import os, io, glob, wave, numpy as np, torch, torch.nn as nn
import pyarrow.parquet as pq
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
MAP={"neutral":"neu","happiness":"hap","anger":"ang","sadness":"sad"}  # 4-class matching our tokens
LAB=["neu","hap","ang","sad"]; L2I={l:i for i,l in enumerate(LAB)}
BASE="facebook/wav2vec2-base"; SR=16000
OUT="/iopsstor/scratch/cscs/mrohania/models/ser4"

def decode(b):
    w=wave.open(io.BytesIO(b),"rb"); n=w.getnframes(); sr=w.getframerate(); ch=w.getnchannels()
    a=np.frombuffer(w.readframes(n),dtype=np.int16).astype(np.float32)/32768.0
    if ch>1: a=a.reshape(-1,ch).mean(1)
    return a, sr

def load_rows():
    X=[]; Y=[]
    for pf in sorted(glob.glob("/iopsstor/scratch/cscs/mrohania/datasets/ser_english/data/*.parquet")):
        t=pq.read_table(pf, columns=["audio","emotion"]).to_pylist()
        for r in t:
            e=MAP.get(str(r["emotion"]).lower());
            if e is None: continue
            try:
                a,sr=decode(r["audio"]["bytes"])
                if sr!=SR or len(a)<0.3*SR or len(a)>12*SR: continue
                X.append(a); Y.append(L2I[e])
            except Exception: continue
    return X,Y

def main():
    dev="cuda"
    fe=AutoFeatureExtractor.from_pretrained(BASE)
    model=Wav2Vec2ForSequenceClassification.from_pretrained(BASE, num_labels=4).to(dev)
    X,Y=load_rows(); print("samples:",len(X),"dist:",{l:Y.count(i) for l,i in L2I.items()}, flush=True)
    idx=np.arange(len(X)); rng=np.random.RandomState(0); rng.shuffle(idx)
    ntr=int(len(idx)*0.9); tr,va=idx[:ntr],idx[ntr:]
    opt=torch.optim.AdamW(model.parameters(), lr=1e-5); B=16
    for ep in range(3):
        model.train(); rng.shuffle(tr)
        for i in range(0,len(tr),B):
            bi=tr[i:i+B]; wavs=[X[j] for j in bi]; ys=torch.tensor([Y[j] for j in bi]).to(dev)
            inp=fe(wavs, sampling_rate=SR, return_tensors="pt", padding=True).input_values.to(dev)
            loss=model(inp, labels=ys).loss; opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval(); cor=0; tot=0
        with torch.no_grad():
            for i in range(0,len(va),B):
                bi=va[i:i+B]; wavs=[X[j] for j in bi]; ys=[Y[j] for j in bi]
                inp=fe(wavs, sampling_rate=SR, return_tensors="pt", padding=True).input_values.to(dev)
                pr=model(inp).logits.argmax(-1).cpu().tolist()
                cor+=sum(int(p==y) for p,y in zip(pr,ys)); tot+=len(ys)
        print(f"epoch {ep}: val acc={cor/max(tot,1):.3f} ({tot})", flush=True)
    os.makedirs(OUT, exist_ok=True); model.save_pretrained(OUT); fe.save_pretrained(OUT)
    print("saved SER4 to", OUT)
if __name__=="__main__": main()
