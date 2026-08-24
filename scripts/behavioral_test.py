"""Behavioral full-duplex test: feed a CONTINUOUS multi-turn user stream (no per-turn reset),
let Moshi converse. Save stereo (user+Moshi), report RTF + Moshi active fraction + turn responsiveness."""
import sys, os, glob, json, time; sys.path.insert(0,"/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import numpy as np, torch, sphn, argparse
from moshi.models import loaders
from moshi.run_inference import InferenceState
def active_frac(pcm, sr, hop=0.08, thr=0.01):
    n=int(sr*hop); fr=[pcm[i:i+n] for i in range(0,max(len(pcm)-n,1),n)]
    return float(np.mean([np.sqrt((f**2).mean()+1e-9)>thr for f in fr])) if fr else 0.0
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--adapter", default=""); ap.add_argument("--config", default=""); ap.add_argument("--label", required=True)
    ap.add_argument("--emotion", default="neu"); ap.add_argument("--secs", type=float, default=24.0)
    a=ap.parse_args(); dev="cuda"
    if a.adapter:
        ci=loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",lora_weights=a.adapter,config_path=a.config)
        lm=ci.get_moshi(device=dev,dtype=torch.bfloat16,fuse_lora=True)
    else:
        ci=loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16"); lm=ci.get_moshi(device=dev,dtype=torch.bfloat16)
    mimi=ci.get_mimi(device=dev); tok=ci.get_text_tokenizer(); sr=mimi.sample_rate
    if getattr(lm,"fuser",None) is not None: os.environ["COND_EMOTION"]=a.emotion
    state=InferenceState(ci,mimi,tok,lm,1,1.0,dev,**ci.lm_gen_config)
    # pick a real multi-turn DailyTalk conversation; use its USER channel (ch1) as continuous input
    base="/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkContiguous/data_stereo"
    wf=sorted(glob.glob(base+"/*.wav"))[-7]
    d,_=sphn.read(wf,sample_rate=sr); user=d[1][:int(a.secs*sr)].astype(np.float32)
    state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
    torch.cuda.synchronize(); t0=time.time()
    out=state.run(torch.from_numpy(user[None,None]).to(dev))
    torch.cuda.synchronize(); gen=time.time()-t0
    moshi=out[0][1][0].detach().cpu().numpy().astype(np.float32)
    L=min(len(user),len(moshi)); stereo=np.stack([moshi[:L],user[:L]])  # ch0 moshi, ch1 user
    outdir="/iopsstor/scratch/cscs/mrohania/audio_samples/behavioral"; os.makedirs(outdir,exist_ok=True)
    sphn.write_wav(f"{outdir}/{a.label}_convo.wav", stereo, sr)
    sphn.write_wav(f"{outdir}/{a.label}_moshi.wav", moshi[None], sr)
    dur=len(user)/sr
    print("BEHAV %-10s RTF=%.3f moshi_active=%.3f user_active=%.3f dur=%.1fs (%s)"%(
        a.label, gen/dur, active_frac(moshi,sr), active_frac(user,sr), dur, a.emotion))
if __name__=="__main__": main()
