"""Affect-congruence A/B: does the emotion CONDITION shift the affect of S_cond_v3's
generated response? For fixed user prompts, generate under COND_EMOTION in {neu,hap,ang,sad},
score each response with dimensional SER (audeering arousal/dominance/valence), compare."""
import sys, os, glob, json; sys.path.insert(0, "/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import numpy as np, torch, sphn, argparse
from moshi.models import loaders
from moshi.run_inference import InferenceState, get_condition_tensors
import ser_dim

def speech_fraction(pcm, sr, hop=0.02, thr=0.01):
    n=int(sr*hop);
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
    ap.add_argument("--adapter", required=True); ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cfg", type=float, default=1.0)
    a=ap.parse_args(); dev="cuda"
    cfg=os.path.join(os.path.dirname(a.adapter),"config.json")
    ci=loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",lora_weights=a.adapter,config_path=cfg)
    lm=ci.get_moshi(device=dev,dtype=torch.bfloat16,fuse_lora=True)
    mimi=ci.get_mimi(device=dev); tok=ci.get_text_tokenizer()
    proc,ser=ser_dim.load(dev)
    prompts=pick_prompts(mimi,a.n); sr=mimi.sample_rate
    outdir="/iopsstor/scratch/cscs/mrohania/audio_samples/affect_ab"; os.makedirs(outdir,exist_ok=True)
    print(f"prompts={len(prompts)}")
    state=InferenceState(ci,mimi,tok,lm,1,a.cfg,dev,**ci.lm_gen_config)   # build once, cfg_coef=a.cfg
    rows={}
    for E in ["neu","hap","ang","sad"]:
        os.environ["COND_EMOTION"]=E
        state.lm_gen.condition_tensors=get_condition_tensors("moshi",lm,1,a.cfg)  # swap condition (cfg-matched)
        segs=[]
        for i,(bn,seg) in enumerate(prompts):
            state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
            out=state.run(torch.from_numpy(seg[None,None]).to(dev))
            rpcm=out[0][1][0].detach().cpu().numpy().astype(np.float32)
            if speech_fraction(rpcm,sr) < 0.05: continue          # skip near-silent responses
            r16=sphn.resample(rpcm, sr, 16000)
            segs.append(np.asarray(r16,dtype=np.float32))
            if i<2: sphn.write_wav(f"{outdir}/p{i}_{E}.wav", rpcm[None], sr)
        if segs:
            vad=np.array(ser_dim.predict(proc,ser,dev,segs))   # [n,3] = aro,dom,val
            rows[E]=(vad.mean(0), len(segs))
        else:
            rows[E]=(np.array([np.nan]*3), 0)
    print("\n=== AFFECT-CONGRUENCE cfg=%.1f (S_cond_v3, SER of generated response) ==="%a.cfg)
    print("CFGRESULT cfg=%.1f"%a.cfg)
    print("%-5s %8s %8s %8s   n" % ("cond","arousal","dominance","valence"))
    for E in ["sad","neu","hap","ang"]:
        m,nn=rows[E]; print("CFGROW cfg=%.1f %-5s %8.3f %8.3f %8.3f   %d"%(a.cfg,E,m[0],m[1],m[2],nn))
    # congruence checks (audeering: index0=arousal,1=dominance,2=valence)
    def g(E,i): return rows[E][0][i]
    print("\nCHECKS (expected directions):")
    print("  valence hap>neu>sad : %.3f > %.3f > %.3f  -> %s"%(g("hap",2),g("neu",2),g("sad",2),
          "PASS" if g("hap",2)>g("neu",2)>g("sad",2) else "partial/FAIL"))
    print("  arousal ang>sad     : %.3f > %.3f       -> %s"%(g("ang",0),g("sad",0),
          "PASS" if g("ang",0)>g("sad",0) else "FAIL"))
    print("  arousal hap>sad     : %.3f > %.3f       -> %s"%(g("hap",0),g("sad",0),
          "PASS" if g("hap",0)>g("sad",0) else "FAIL"))
    print("  dominance ang>sad   : %.3f > %.3f       -> %s"%(g("ang",1),g("sad",1),
          "PASS" if g("ang",1)>g("sad",1) else "FAIL"))

if __name__=="__main__": main()
