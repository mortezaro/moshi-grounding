"""Natural-clip perception check: run SER on IEMOCAP *improvised* clips (real human emotion),
compare perceived vs true, and report VAD ranges to calibrate thresholds."""
import sys, glob, tempfile, os, collections; sys.path.insert(0,"/iopsstor/scratch/cscs/mrohania/moshi-finetune")
import numpy as np, sphn, pyarrow.parquet as pq
import ser_dim
EMAP={"hap":"hap","exc":"hap","sad":"sad","ang":"ang","fru":"ang","neu":"neu"}
def main():
    dev="cuda"; proc,ser=ser_dim.load(dev)
    fs=[f for f in sorted(glob.glob("/iopsstor/scratch/cscs/mrohania/hf_cache/datasets--Ar4ikov--iemocap_audio_text_splitted/snapshots/*/**/*.parquet", recursive=True)) if "test-" not in f]
    by=collections.defaultdict(list)
    for f in fs:
        for r in pq.read_table(f,columns=["titre","emotion","audio"]).to_pylist():
            if "impro" not in (r["titre"] or ""): continue          # natural / improvised only
            e=EMAP.get(r["emotion"]);
            if e and len(by[e])<25: by[e].append(r)
        if all(len(by[e])>=25 for e in ["hap","sad","ang","neu"]): break
    def vadof(b):
        with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as tf: tf.write(b); tp=tf.name
        try: d,s=sphn.read(tp)
        finally: os.unlink(tp)
        w=d.mean(0) if d.ndim>1 else d
        w16=np.asarray(sphn.resample(w,s,16000),dtype=np.float32) if s!=16000 else w
        return np.array(ser_dim.predict(proc,ser,dev,[w16]))[0]
    # collect VAD per true emotion
    stats=collections.defaultdict(list)
    for e in ["hap","sad","ang","neu"]:
        for r in by[e][:20]:
            try: stats[e].append(vadof(r["audio"]["bytes"]))
            except Exception: pass
    print("=== SER VAD by true emotion (natural IEMOCAP improvised) ===")
    allv=[]
    for e in ["hap","sad","ang","neu"]:
        arr=np.array(stats[e]); allv+=list(arr)
        print("  %-4s n=%d  arousal=%.2f dominance=%.2f valence=%.2f"%(e,len(arr),arr[:,0].mean(),arr[:,1].mean(),arr[:,2].mean()))
    allv=np.array(allv)
    # calibrated thresholds from data percentiles
    aL,aH=np.percentile(allv[:,0],[40,60]); vL,vH=np.percentile(allv[:,2],[40,60])
    print("calibrated thresholds: aro[%.2f,%.2f] val[%.2f,%.2f]"%(aL,aH,vL,vH))
    def perc(a,d,v):
        if a>aH and v<vL: return "ang"
        if v>vH: return "hap"
        if v<vL: return "sad"
        return "neu"
    print("=== perception accuracy (calibrated) ===")
    conf=collections.Counter(); correct=0; tot=0
    for e in ["hap","sad","ang","neu"]:
        for vad in stats[e]:
            p=perc(*vad); conf[(e,p)]+=1; tot+=1; correct+= (p==e)
    print("  overall acc=%.0f%% (n=%d)"%(100*correct/max(tot,1),tot))
    for e in ["hap","sad","ang","neu"]:
        row=" ".join("%s=%d"%(p,conf[(e,p)]) for p in ["hap","sad","ang","neu"] if conf[(e,p)])
        print("  true %-4s -> %s"%(e,row))
if __name__=="__main__": main()
