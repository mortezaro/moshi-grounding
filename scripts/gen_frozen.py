"""Generate from a FULL-MODEL checkpoint (base+conditioner, frozen-base run) or base.
Same prompts, COND_EMOTION fixed. Saves wavs for A/B vs base."""
import sys, os, glob, json; sys.path.insert(0,"/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import numpy as np, torch, sphn, argparse
from moshi.models import loaders
from moshi.run_inference import InferenceState
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
    ap.add_argument("--weights", default=""); ap.add_argument("--config", default=""); ap.add_argument("--adapter", default="")
    ap.add_argument("--label", required=True); ap.add_argument("--emotion", default="neu"); ap.add_argument("--n", type=int, default=3)
    a=ap.parse_args(); dev="cuda"
    if a.adapter:
        ci=loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16", lora_weights=a.adapter, config_path=a.config)
        lm=ci.get_moshi(device=dev,dtype=torch.bfloat16,fuse_lora=True)
    elif a.weights:
        ci=loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16", moshi_weights=a.weights, config_path=a.config)
    else:
        ci=loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
    lm=lm if a.adapter else ci.get_moshi(device=dev,dtype=torch.bfloat16)
    mimi=ci.get_mimi(device=dev); tok=ci.get_text_tokenizer()
    os.environ["COND_EMOTION"]=a.emotion
    state=InferenceState(ci,mimi,tok,lm,1,1.0,dev,**ci.lm_gen_config)
    outdir="/iopsstor/scratch/cscs/mrohania/audio_samples/frozen_ab"; os.makedirs(outdir,exist_ok=True)
    prompts=pick_prompts(mimi,a.n); sr=mimi.sample_rate
    for i,(bn,seg) in enumerate(prompts):
        if a.label=="base": sphn.write_wav(f"{outdir}/p{i}_input.wav", seg[None].astype(np.float32), sr)
        state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
        out=state.run(torch.from_numpy(seg[None,None]).to(dev))
        rpcm=out[0][1][0].detach().cpu().numpy().astype(np.float32)
        sphn.write_wav(f"{outdir}/p{i}_{a.label}.wav", rpcm[None], sr)
    print("SAVED %s (emotion=%s)"%(a.label,a.emotion))
if __name__=="__main__": main()
