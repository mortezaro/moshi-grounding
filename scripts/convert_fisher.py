#!/usr/bin/env python
"""Convert kalbin/fisher-sft-v2 parquet -> moshi-finetune contiguous-stereo format,
STITCHING N turn-pairs into one multi-turn clip (efficient: less padding, denser grounding).
Each pair: user(prompt) on R, then Moshi(completion) on L after a gap. Pairs are laid
end-to-end. Moshi utterances become SPEAKER_MAIN word-alignments (uniform within turn)."""
import os, io, json, glob, wave, argparse, numpy as np, sphn
SR=16000; GAP=0.4; MINDUR=0.4; TURNGAP=0.5

def decode_wav(b):
    w=wave.open(io.BytesIO(b),"rb"); n=w.getnframes(); sr=w.getframerate(); ch=w.getnchannels()
    a=np.frombuffer(w.readframes(n),dtype=np.int16).astype(np.float32)/32768.0
    if ch>1: a=a.reshape(-1,ch).mean(1)
    if sr!=SR and n>0: a=sphn.resample(a[None], src_sample_rate=sr, dst_sample_rate=SR)[0]
    return a

def build_clip(pairs):
    """pairs: list of (user_audio, moshi_audio, moshi_text). Returns (L,R,alignments)."""
    segsL=[]; segsR=[]; al=[]; cursor=0.0
    for pa,ca,ct in pairs:
        ud=len(pa)/SR; md=len(ca)/SR
        # user turn on R
        segsL.append(np.zeros(len(pa),np.float32)); segsR.append(pa); ustart=cursor; cursor+=ud
        # gap
        g=int(GAP*SR); segsL.append(np.zeros(g,np.float32)); segsR.append(np.zeros(g,np.float32)); cursor+=GAP
        # moshi turn on L
        segsL.append(ca); segsR.append(np.zeros(len(ca),np.float32)); mstart=cursor
        ws=[w for w in ct.strip().split() if w]
        if ws:
            step=md/len(ws)
            for i,w in enumerate(ws):
                al.append([w,[round(mstart+i*step,3),round(mstart+(i+1)*step,3)],"SPEAKER_MAIN"])
        cursor+=md
        # inter-turn gap
        tg=int(TURNGAP*SR); segsL.append(np.zeros(tg,np.float32)); segsR.append(np.zeros(tg,np.float32)); cursor+=TURNGAP
    return np.concatenate(segsL), np.concatenate(segsR), al

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", default="/iopsstor/scratch/cscs/mrohania/datasets/fisher_sft_v2/data")
    ap.add_argument("--out", default="/iopsstor/scratch/cscs/mrohania/datasets/FisherConv")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--stitch", type=int, default=12)
    a=ap.parse_args()
    import pyarrow.parquet as pq
    ow=os.path.join(a.out,"data_stereo"); os.makedirs(ow,exist_ok=True)
    files=sorted(glob.glob(a.src+"/*.parquet"))
    mine=[f for i,f in enumerate(files) if i%a.nshards==a.shard]
    jl=open(os.path.join(a.out,f"shard_{a.shard}.jsonl"),"w"); n=0; skip=0
    for pf in mine:
        try: t=pq.read_table(pf, columns=["completion_audio","prompt_audio","completion_text"])
        except Exception as ex: print("read err",pf,ex); continue
        rows=t.to_pylist(); buf=[]
        for row in rows:
            try:
                pa=decode_wav(row["prompt_audio"]["bytes"]); ca=decode_wav(row["completion_audio"]["bytes"])
                ct=row["completion_text"] or ""
                if len(ca)<MINDUR*SR or len(pa)<MINDUR*SR or not ct.strip(): skip+=1; continue
                buf.append((pa,ca,ct))
            except Exception: skip+=1; continue
            if len(buf)>=a.stitch:
                L,R,al=build_clip(buf); buf=[]
                if not al: continue
                bn=f"{os.path.basename(pf)[:-8]}_{n}"
                sphn.write_wav(os.path.join(ow,bn+".wav"), np.stack([L,R]), SR)
                json.dump({"alignments":al}, open(os.path.join(ow,bn+".json"),"w"))
                jl.write(json.dumps({"path":f"data_stereo/{bn}.wav","duration":len(L)/SR})+"\n"); n+=1
        print(f"[shard {a.shard}] {os.path.basename(pf)}: clips {n} skip {skip}", flush=True)
    jl.close(); print(f"[shard {a.shard}] DONE clips={n} skip={skip}")
if __name__=="__main__": main()
