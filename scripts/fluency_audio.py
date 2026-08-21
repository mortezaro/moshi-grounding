#!/usr/bin/env python
"""Better fluency metric + audio samples. Feeds FIXED user prompts, generates response
audio, measures SPEECH-FRACTION (frac of output frames with speech energy, not binary),
and SAVES wavs (input prompt + response) for listening."""
import sys, os, glob, json; sys.path.insert(0,"/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import numpy as np, torch, sphn, argparse
from moshi.models import loaders
from moshi.run_inference import InferenceState
def speech_fraction(pcm, sr, hop=0.02, thr=0.01):
    n=int(hop*sr); m=len(pcm)//n
    if m==0: return 0.0
    fr=np.sqrt((pcm[:m*n].reshape(m,n)**2).mean(1)+1e-9)
    return float((fr>thr).mean())
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
    ap.add_argument("--adapter", default=""); ap.add_argument("--label", required=True); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cfg", type=float, default=1.0)
    a=ap.parse_args(); dev="cuda"
    if not a.adapter:
        ci=loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16"); lm=ci.get_moshi(device=dev,dtype=torch.bfloat16)
    else:
        cfg=os.path.join(os.path.dirname(a.adapter),"config.json")
        ci=loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",lora_weights=a.adapter,config_path=cfg)
        lm=ci.get_moshi(device=dev,dtype=torch.bfloat16,fuse_lora=True)
    mimi=ci.get_mimi(device=dev); tok=ci.get_text_tokenizer()
    state=InferenceState(ci,mimi,tok,lm,1,a.cfg,dev,**ci.lm_gen_config)
    outdir="/iopsstor/scratch/cscs/mrohania/audio_samples"; os.makedirs(outdir,exist_ok=True)
    prompts=pick_prompts(mimi,a.n); sr=mimi.sample_rate; fracs=[]
    for i,(bn,seg) in enumerate(prompts):
        if a.label=="base" or i==0:  # save the shared input prompt once
            sphn.write_wav(f"{outdir}/prompt{i}_input.wav", seg[None].astype(np.float32), sr)
        state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
        out=state.run(torch.from_numpy(seg[None,None]).to(dev))
        rpcm=out[0][1][0].detach().cpu().numpy().astype(np.float32)
        sf=speech_fraction(rpcm,sr); fracs.append(sf)
        sphn.write_wav(f"{outdir}/prompt{i}_{a.label}.wav", rpcm[None], sr)
    print("SPEECHFRAC %-14s mean=%.3f per=%s"%(a.label, np.mean(fracs), [round(x,2) for x in fracs]))
if __name__=="__main__": main()
