"""Axis A/B: set one dimensional condition (arousal|dominance|valence) lo/mid/hi,
measure the SER value of that same axis in the response. Tests independent multi-dim control."""
import sys, os, glob, json; sys.path.insert(0, "/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import numpy as np, torch, sphn, argparse
from moshi.models import loaders
from moshi.run_inference import InferenceState, get_condition_tensors
import ser_dim
IDX={"arousal":0,"dominance":1,"valence":2}
def speech_fraction(pcm, sr, hop=0.02, thr=0.01):
    n=int(sr*hop)
    if len(pcm)<n: return 0.0
    fr=[pcm[i:i+n] for i in range(0,len(pcm)-n,n)]
    return float(np.mean([np.sqrt((f**2).mean()+1e-9)>thr for f in fr]))
def pick_prompts(mimi, n):
    base="/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkContiguous/data_stereo"
    out=[]
    for wf in sorted(glob.glob(base+"/*.wav"))[-200:]:
        al=json.load(open(wf[:-4]+".json"))["alignments"]; d,_=sphn.read(wf,sample_rate=mimi.sample_rate)
        cur=None; mt=[]
        for w,ts,s in al:
            if cur is None: cur=[ts[0],ts[1]]
            elif ts[0]-cur[1]<=0.6: cur[1]=ts[1]
            else: mt.append(tuple(cur)); cur=[ts[0],ts[1]]
        if cur: mt.append(tuple(cur))
        prev=0.0
        for t0,t1 in mt:
            if t0-prev>=0.5:
                seg=d[1][int(prev*mimi.sample_rate):int(t0*mimi.sample_rate)]
                if np.sqrt((seg**2).mean()+1e-9)>0.02 and (t0-prev)>1.5: out.append((os.path.basename(wf),seg))
            prev=t1
            if len(out)>=n: return out
    return out
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True); ap.add_argument("--axis", required=True, choices=list(IDX))
    ap.add_argument("--n", type=int, default=14); ap.add_argument("--cfg", type=float, default=1.0)
    a=ap.parse_args(); dev="cuda"; ix=IDX[a.axis]
    cfg=os.path.join(os.path.dirname(a.adapter),"config.json")
    ci=loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",lora_weights=a.adapter,config_path=cfg)
    lm=ci.get_moshi(device=dev,dtype=torch.bfloat16,fuse_lora=True)
    mimi=ci.get_mimi(device=dev); tok=ci.get_text_tokenizer(); proc,ser=ser_dim.load(dev)
    prompts=pick_prompts(mimi,a.n); sr=mimi.sample_rate
    os.environ["COND_EMOTION"]="neu"
    state=InferenceState(ci,mimi,tok,lm,1,a.cfg,dev,**ci.lm_gen_config)
    rows={}
    for V in ["lo","mid","hi"]:
        os.environ["COND_"+a.axis.upper()]=V
        state.lm_gen.condition_tensors=get_condition_tensors("moshi",lm,1,a.cfg)
        segs=[]
        for bn,seg in prompts:
            state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
            out=state.run(torch.from_numpy(seg[None,None]).to(dev))
            rpcm=out[0][1][0].detach().cpu().numpy().astype(np.float32)
            if speech_fraction(rpcm,sr)<0.05: continue
            segs.append(np.asarray(sphn.resample(rpcm,sr,16000),dtype=np.float32))
        vad=np.array(ser_dim.predict(proc,ser,dev,segs)) if segs else np.full((1,3),np.nan)
        rows[V]=(vad.mean(0), len(segs))
    print("\n=== %s STEERING (cond %s -> SER %s of response) ==="%(a.axis.upper(),a.axis,a.axis))
    for V in ["lo","mid","hi"]:
        m,nn=rows[V]; print("AXROW %-8s %-4s val=%.3f n=%d"%(a.axis,V,m[ix],nn))
    lo,mid,hi=rows["lo"][0][ix],rows["mid"][0][ix],rows["hi"][0][ix]
    print("AXCHECK %s hi>mid>lo: %.3f>%.3f>%.3f -> %s (spread %.3f)"%(a.axis,hi,mid,lo,"PASS" if hi>mid>lo else ("DIR-OK" if hi>lo else "FAIL"),hi-lo))
if __name__=="__main__": main()
