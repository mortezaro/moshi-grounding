#!/usr/bin/env python
"""Train a 4-class emotion classifier on MELD (conversational dialogue) — domain-matched
to our DailyTalk/Fisher data, unlike acted IEMOCAP. Maps MELD 7-class ints to neu/hap/ang/sad."""
import os, io, glob, wave, numpy as np, torch
import pyarrow.parquet as pq
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
# standard MELD order: 0 anger,1 disgust,2 fear,3 joy,4 neutral,5 sadness,6 surprise
MELD2OUR={0:"ang",3:"hap",4:"neu",5:"sad"}
LAB=["neu","hap","ang","sad"]; L2I={l:i for i,l in enumerate(LAB)}
BASE="facebook/wav2vec2-base"; SR=16000
OUT="/iopsstor/scratch/cscs/mrohania/models/meld4"
def decode(b):
    w=wave.open(io.BytesIO(b),"rb"); n=w.getnframes(); sr=w.getframerate(); ch=w.getnchannels()
    a=np.frombuffer(w.readframes(n),dtype=np.int16).astype(np.float32)/32768.0
    if ch>1: a=a.reshape(-1,ch).mean(1)
    return a,sr
def load(split):
    X=[];Y=[]
    for pf in sorted(glob.glob("/iopsstor/scratch/cscs/mrohania/datasets/meld_audio/data/*.parquet")):
        if split not in os.path.basename(pf): continue
        for r in pq.read_table(pf,columns=["audio","label"]).to_pylist():
            e=MELD2OUR.get(int(r["label"]))
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
    model=Wav2Vec2ForSequenceClassification.from_pretrained(BASE,num_labels=4).to(dev)
    Xtr,Ytr=load("train"); Xva,Yva=load("validation")
    print("train:",len(Xtr),{l:Ytr.count(i) for l,i in L2I.items()},"| val:",len(Xva),flush=True)
    # class-weighted loss to counter neutral dominance
    import collections; cnt=collections.Counter(Ytr); tot=sum(cnt.values())
    w=torch.tensor([tot/(4*cnt.get(i,1)) for i in range(4)],device=dev,dtype=torch.float)
    lossf=torch.nn.CrossEntropyLoss(weight=w)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5); B=16; rng=np.random.RandomState(0)
    idx=np.arange(len(Xtr))
    for ep in range(4):
        model.train(); rng.shuffle(idx)
        for i in range(0,len(idx),B):
            bi=idx[i:i+B]; wavs=[Xtr[j] for j in bi]; ys=torch.tensor([Ytr[j] for j in bi]).to(dev)
            inp=fe(wavs,sampling_rate=SR,return_tensors="pt",padding=True).input_values.to(dev)
            loss=lossf(model(inp).logits,ys); opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); from collections import Counter as C; hit=C();tc=C()
        with torch.no_grad():
            for i in range(0,len(Xva),B):
                wavs=Xva[i:i+B]; ys=Yva[i:i+B]
                inp=fe(wavs,sampling_rate=SR,return_tensors="pt",padding=True).input_values.to(dev)
                pr=model(inp).logits.argmax(-1).cpu().tolist()
                for p,y in zip(pr,ys): tc[y]+=1; hit[y]+=int(p==y)
        bal=sum(hit[c]/tc[c] for c in tc)/len(tc); raw=sum(hit.values())/max(sum(tc.values()),1)
        print("epoch %d: val raw=%.3f bal=%.3f"%(ep,raw,bal),flush=True)
    os.makedirs(OUT,exist_ok=True); model.save_pretrained(OUT); fe.save_pretrained(OUT)
    print("saved MELD4 to",OUT)
if __name__=="__main__": main()
