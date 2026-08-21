import io, json, os, wave, numpy as np, sphn
import pyarrow.parquet as pq
SR=16000; OUT="/iopsstor/scratch/cscs/mrohania/datasets/LibriOOD"; os.makedirs(OUT+"/data_stereo",exist_ok=True)
def dec(b):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".flac",delete=False) as f: f.write(b); tp=f.name
    d,sr=sphn.read(tp,sample_rate=SR); os.unlink(tp)
    return d[0] if d.ndim>1 else d
t=pq.read_table("/iopsstor/scratch/cscs/mrohania/datasets/librispeech/all/test.clean/0000.parquet")
cols=t.column_names; print("cols:",cols)
rows=t.slice(0,50).to_pylist()
jl=open(OUT+"/libri.jsonl","w"); n=0
for i,r in enumerate(rows):
    au=r.get("audio"); txt=(r.get("text") or "").strip()
    if not au or not txt: continue
    a=dec(au["bytes"]); dur=len(a)/SR
    if dur<1 or dur>30: continue
    L=a; R=np.zeros_like(a)
    ws=txt.split(); step=dur/max(len(ws),1)
    al=[[w,[round(j*step,3),round((j+1)*step,3)],"SPEAKER_MAIN"] for j,w in enumerate(ws)]
    bn=f"libri_{i}"
    sphn.write_wav(OUT+f"/data_stereo/{bn}.wav", np.stack([L,R]), SR)
    json.dump({"alignments":al}, open(OUT+f"/data_stereo/{bn}.json","w"))
    jl.write(json.dumps({"path":f"data_stereo/{bn}.wav","duration":dur})+"\n"); n+=1
jl.close(); print("LibriOOD clips:",n)
