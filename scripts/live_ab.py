"""Live full-duplex A/B on S_cond_val100: stream the SAME multi-turn user input under
COND_VALENCE lo vs hi. Reports RTF (real-time?), speech-fraction (turn-taking alive?),
SER valence of the streamed response, and saves wavs to listen to."""
import sys, os, glob, json, time; sys.path.insert(0,"/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import numpy as np, torch, sphn, argparse
from moshi.models import loaders
from moshi.run_inference import InferenceState, get_condition_tensors
import ser_dim
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
    ap.add_argument("--adapter", required=True); ap.add_argument("--n", type=int, default=16)
    a=ap.parse_args(); dev="cuda"
    cfg=os.path.join(os.path.dirname(a.adapter),"config.json")
    ci=loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",lora_weights=a.adapter,config_path=cfg)
    lm=ci.get_moshi(device=dev,dtype=torch.bfloat16,fuse_lora=True)
    mimi=ci.get_mimi(device=dev); tok=ci.get_text_tokenizer(); proc,ser=ser_dim.load(dev)
    prompts=pick_prompts(mimi,a.n); sr=mimi.sample_rate
    outdir="/iopsstor/scratch/cscs/mrohania/audio_samples/live_ab"; os.makedirs(outdir,exist_ok=True)
    os.environ["COND_EMOTION"]="neu"
    state=InferenceState(ci,mimi,tok,lm,1,1.0,dev,**ci.lm_gen_config)
    print("prompts=%d  (RTF = gen_time/input_dur; <=1 is real-time)"%len(prompts))
    for V in ["lo","hi"]:
        os.environ["COND_VALENCE"]=V
        state.lm_gen.condition_tensors=get_condition_tensors("moshi",lm,1,1.0)
        rtfs=[]; fracs=[]; segs=[]; saved=0
        for i,(bn,seg) in enumerate(prompts):
            state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
            dur=seg.shape[0]/sr
            torch.cuda.synchronize(); t0=time.time()
            out=state.run(torch.from_numpy(seg[None,None]).to(dev))
            torch.cuda.synchronize(); gen=time.time()-t0
            rtfs.append(gen/max(dur,1e-3))
            rpcm=out[0][1][0].detach().cpu().numpy().astype(np.float32)
            sf=speech_fraction(rpcm,sr); fracs.append(sf)
            if sf>=0.05: segs.append(np.asarray(sphn.resample(rpcm,sr,16000),dtype=np.float32))
            if saved<3 and sf>=0.05:
                sphn.write_wav(f"{outdir}/turn{i}_{V}.wav", rpcm[None], sr)
                if V=="lo": sphn.write_wav(f"{outdir}/turn{i}_userinput.wav", seg[None].astype(np.float32), sr)
                saved+=1
        vad=np.array(ser_dim.predict(proc,ser,dev,segs)) if segs else np.full((1,3),np.nan)
        import statistics as st
        talk=float(np.mean([f>=0.05 for f in fracs]))
        print("VAL=%s | RTF mean=%.3f median=%.3f max=%.3f | speech-frac=%.3f | turns-with-speech=%.0f%% | resp valence=%.3f (n=%d)"%(
            V, st.mean(rtfs), st.median(rtfs), max(rtfs), np.mean(fracs), 100*talk, vad.mean(0)[2], len(segs)))
    print("wavs -> %s (turnN_lo.wav vs turnN_hi.wav, + userinput)"%outdir)
if __name__=="__main__": main()
