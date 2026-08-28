#!/usr/bin/env python
"""Convert kalbin/moshi-on-policy-dpo-9k -> our DPO pref format (stereo wav + alignments).

The dataset is Moshi-mono only (no user audio): chosen/rejected are ~30s Moshi responses,
user side is text. We build, per row, two stereo wavs (ch0 = trimmed Moshi response, ch1 =
silent) + .json alignments from chosen_text / rejected_text (uniform word timing, as
build_expresso does), and a prefs jsonl with {chosen, rejected, duration}. dpo_train.py
then consumes this unchanged.
"""
import os, re, json, argparse, tempfile
import numpy as np, sphn
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
SR = 24000; REPO = "kalbin/moshi-on-policy-dpo-9k"

def decode(cell):
    b = cell.get("bytes") if isinstance(cell, dict) else None
    if b is None: return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf: tf.write(b); tp = tf.name
    try: d, sr = sphn.read(tp)
    finally: os.unlink(tp)
    w = d.mean(0) if d.ndim > 1 else d
    if sr != SR: w = sphn.resample(w, sr, SR)
    return np.asarray(w, dtype=np.float32)

def trim(w, thr=0.008, pad=0.3, maxs=18.0):
    n = int(0.02*SR); m = len(w)//n
    if m == 0: return w
    fr = np.sqrt((w[:m*n].reshape(m, n)**2).mean(1)+1e-9)
    voiced = np.where(fr > thr)[0]
    if len(voiced) == 0: return w[:int(1.0*SR)]
    end = min(len(w), int((voiced[-1]+1)*n + pad*SR))
    start = max(0, int(voiced[0]*n - pad*SR))
    return w[start:min(end, start+int(maxs*SR))]

def aligns_from_text(txt, dur):
    words = re.findall(r"[A-Za-z']+|[.?!,]", txt.strip())
    words = [w for w in words if re.search(r"[A-Za-z]", w)]
    if not words or dur <= 0: return []
    dt = dur/len(words)
    return [[w, [round(i*dt, 3), round((i+1)*dt, 3)], "SPEAKER_MAIN"] for i, w in enumerate(words)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--shards", type=int, default=15)
    ap.add_argument("--min_margin", type=float, default=1.0); ap.add_argument("--shard_id", type=int, default=0)
    a = ap.parse_args()
    ow = os.path.join(a.out, "data_stereo"); os.makedirs(ow, exist_ok=True)
    api = HfApi(); info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)
    sib = sorted([s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet")])
    sib = sib[a.shard_id::max(1, 108//a.shards)][:a.shards]     # spread across the 108 shards
    jo = open(os.path.join(a.out, f"prefs_{a.shard_id}.jsonl"), "w"); npair = 0
    for si, fn in enumerate(sib):
        p = hf_hub_download(REPO, fn, repo_type="dataset")
        for r in pq.read_table(p, columns=["chosen","rejected","chosen_text","rejected_text",
                                           "chosen_score","rejected_score","score_margin"]).to_pylist():
            if (r.get("score_margin") or 0) < a.min_margin: continue
            wc = decode(r["chosen"]); wr = decode(r["rejected"])
            if wc is None or wr is None: continue
            wc = trim(wc); wr = trim(wr)
            ct = r.get("chosen_text") or ""; rt = r.get("rejected_text") or ""
            ac = aligns_from_text(ct, len(wc)/SR); ar = aligns_from_text(rt, len(wr)/SR)
            if not ac or not ar: continue
            pid = f"{a.shard_id}_{npair}"; paths = {}
            for tag, w, al in (("c", wc, ac), ("r", wr, ar)):
                buf = np.zeros((2, len(w)), dtype=np.float32); buf[0] = w
                pk = np.abs(buf).max()
                if pk > 0: buf = buf*(0.95/pk)
                bn = f"{pid}_{tag}"; sphn.write_wav(os.path.join(ow, bn+".wav"), buf, SR)
                json.dump({"alignments": al, "text_conditions": {"emotion": "neu"}},
                          open(os.path.join(ow, bn+".json"), "w"))
                paths[tag] = (f"data_stereo/{bn}.wav", len(w)/SR)
            jo.write(json.dumps({"chosen": paths["c"][0], "rejected": paths["r"][0],
                                 "duration": max(paths["c"][1], paths["r"][1]),
                                 "margin": r.get("score_margin")})+"\n")
            npair += 1
        print(f"[shard {si+1}/{len(sib)}] {fn.split('/')[-1]} -> total pairs={npair}", flush=True)
    jo.close(); print(f"DONE converted pairs={npair}", flush=True)

if __name__ == "__main__":
    main()
