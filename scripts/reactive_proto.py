"""Phase-1 reactive prototype (offline): emotional user clip -> SER perceives -> policy picks
congruent response emotion -> Moshi generates its reply under that emotion. Proves the loop."""
import sys, os, glob, json, tempfile; sys.path.insert(0,"/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import numpy as np, torch, sphn, argparse, pyarrow.parquet as pq
from moshi.models import loaders
from moshi.run_inference import InferenceState, get_condition_tensors
import ser_dim
# perceive: dimensional VAD -> categorical (audeering thresholds, [0,1] scale)
def perceive(a, d, v):
    ah, al = a>0.55, a<0.42; vh, vl = v>0.55, v<0.45
    if ah and vl: return "ang"
    if vh: return "hap"
    if vl or (al and vl): return "sad"
    return "neu"
# empathetic policy: mirror positive, empathise with sad, calm anger
POLICY={"hap":"hap","sad":"sad","ang":"neu","neu":"neu"}
def emo_user_clips(n=5):
    fs=sorted(glob.glob("/iopsstor/scratch/cscs/mrohania/hf_cache/datasets--xbgoose--ravdess/snapshots/*/**/*.parquet", recursive=True))
    want=["happy","sad","angry","neutral","calm"]; picked={}
    for f in fs:
        for r in pq.read_table(f,columns=["audio","emotion","emotional_intensity","actor"]).to_pylist():
            e=r["emotion"]
            if e in want and e not in picked and (r["emotional_intensity"]=="strong" or e in ("neutral","calm")):
                picked[e]=r
        if len(picked)>=len(want): break
    return [(e,picked[e]) for e in want if e in picked]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True); ap.add_argument("--config", required=True); ap.add_argument("--label", required=True)
    a=ap.parse_args(); dev="cuda"
    ci=loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",lora_weights=a.adapter,config_path=a.config)
    lm=ci.get_moshi(device=dev,dtype=torch.bfloat16,fuse_lora=True)
    mimi=ci.get_mimi(device=dev); tok=ci.get_text_tokenizer(); proc,ser=ser_dim.load(dev); sr=mimi.sample_rate
    outdir="/iopsstor/scratch/cscs/mrohania/audio_samples/reactive"; os.makedirs(outdir,exist_ok=True)
    def dec(b):
        with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as tf: tf.write(b); tp=tf.name
        try: d,s=sphn.read(tp)
        finally: os.unlink(tp)
        w=d.mean(0) if d.ndim>1 else d
        return w, s
    clips=emo_user_clips()
    os.environ["COND_EMOTION"]="neu"
    state=InferenceState(ci,mimi,tok,lm,1,1.0,dev,**ci.lm_gen_config)   # build ONCE
    print("=== REACTIVE LOOP  model=%s ==="%a.label)
    print("%-9s %-10s %-9s %-9s"%("user(true)","perceived","->respond","(policy)"))
    for true_e, r in clips:
        w48, s = dec(r["audio"]["bytes"])
        w16 = np.asarray(sphn.resample(w48, s, 16000), dtype=np.float32)
        vad = np.array(ser_dim.predict(proc, ser, dev, [w16]))[0]   # [aro,dom,val] in [0,1]
        perc = perceive(vad[0], vad[1], vad[2])
        target = POLICY[perc]
        # generate Moshi response to this user clip, conditioned on target emotion
        os.environ["COND_EMOTION"]=target
        state.lm_gen.condition_tensors=get_condition_tensors("moshi",lm,1,1.0)   # swap condition live
        user_m = np.asarray(sphn.resample(w48, s, sr), dtype=np.float32)
        state.mimi.reset_streaming(); state.lm_gen.reset_streaming()
        out=state.run(torch.from_numpy(user_m[None,None]).to(dev))
        resp=out[0][1][0].detach().cpu().numpy().astype(np.float32)
        sphn.write_wav(f"{outdir}/{a.label}_{true_e}_user.wav", user_m[None], sr)
        sphn.write_wav(f"{outdir}/{a.label}_{true_e}_resp_{target}.wav", resp[None], sr)
        print("%-9s %-10s -> %-6s   VAD=(%.2f,%.2f,%.2f)"%(true_e, perc, target, vad[0],vad[1],vad[2]))
if __name__=="__main__": main()
