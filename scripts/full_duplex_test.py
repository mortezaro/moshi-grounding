#!/usr/bin/env python
"""FULL-DUPLEX validation: run the model in streaming (live) mode.
Measures (1) real-time factor RTF = gen_time/audio_dur (must be <=1 for real-time),
(2) affect-responsiveness: feed user turns of varied emotion, SER the GENERATED response
audio, and check whether the response affect tracks the user affect (better conversation)
more for the grounded model than base."""
import sys, time, os, glob, json; sys.path.insert(0,"/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import numpy as np, torch, sphn, ser_dim
from moshi.models import loaders
from moshi.run_inference import InferenceState
import argparse
_T=json.load(open("/iopsstor/scratch/cscs/mrohania/moshi_emo/thresholds.json"))
def emo(a,vl):
    ah=a>=_T["aro_hi"]; al=a<=_T["aro_lo"]; vh=vl>=_T["val_hi"]; vll=vl<=_T["val_lo"]
    if ah and vll: return "ang"
    if ah and vh: return "hap"
    if al and vll: return "sad"
    if vh: return "hap"
    if vll: return "sad"
    return "neu"
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--adapter", default=""); ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=16)
    a=ap.parse_args(); dev="cuda"
    if not a.adapter:
        ci=loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16"); lm=ci.get_moshi(device=dev,dtype=torch.bfloat16)
    else:
        cfg=os.path.join(os.path.dirname(a.adapter),"config.json")
        ci=loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",lora_weights=a.adapter,config_path=cfg)
        lm=ci.get_moshi(device=dev,dtype=torch.bfloat16,fuse_lora=True)
    mimi=ci.get_mimi(device=dev); tok=ci.get_text_tokenizer()
    proc,sermodel=ser_dim.load(dev)
    state=InferenceState(ci,mimi,tok,lm,1,1.0,dev,**ci.lm_gen_config)
    # gather user turns (right channel gaps) with their SER emotion, spanning emotions
    base="/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkContiguous/data_stereo"
    turns=[]
    for wf in sorted(glob.glob(base+"/*.wav"))[-120:]:
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
                if np.sqrt((seg**2).mean()+1e-9)>0.02 and (t0-prev)>1.0: turns.append(seg)
            prev=t1
        if len(turns)>=a.n: break
    rtfs=[]; pairs=[]
    for seg in turns[:a.n]:
        up=torch.from_numpy(seg[None,None]).to(dev)
        dur=seg.shape[0]/mimi.sample_rate
        state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
        t0=time.time(); out=state.run(up); gen=time.time()-t0
        rtfs.append(gen/max(dur,1e-3))
        # user affect
        va=ser_dim.predict(proc,sermodel,dev,[seg.astype(np.float32)])[0]; ue=emo(va[0],va[2])
        # response audio from audio_tokens -> decode -> SER
        try:
            rpcm=out[0][1]                     # already-decoded PCM [1, T_samples]
            rwav=rpcm[0].detach().cpu().numpy().astype(np.float32)
            if len(rwav)>mimi.sample_rate*0.3 and np.sqrt((rwav**2).mean()+1e-9)>0.005:
                rv=ser_dim.predict(proc,sermodel,dev,[rwav])[0]; re=emo(rv[0],rv[2])
            else: re="(silent)"
        except Exception as ex: re="(err:%s)"%str(ex)[:40]
        pairs.append((ue,re))
    import statistics
    print("LABEL %s | RTF mean=%.3f median=%.3f max=%.3f  (<=1.0 = real-time)"%(a.label,statistics.mean(rtfs),statistics.median(rtfs),max(rtfs)))
    from collections import Counter
    # affect responsiveness: how many DISTINCT response emotions across varied user inputs
    resp=[r for u,r in pairs if not r.startswith("(")]
    print("  user->response affect pairs:", pairs)
    print("  response affect diversity: %d distinct / %d (higher=more responsive)"%(len(set(resp)), len(resp)))
if __name__=="__main__": main()
