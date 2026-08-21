"""Reconstruct dyadic multi-turn stereo from IEMOCAP (Ar4ikov mirror) for Moshi conditioning.
Real timeline (start/end) -> true full-duplex stereo (overlaps/pauses preserved).
Human VAD labels -> text_conditions {emotion, arousal, dominance} (real, not SER pseudo).
Output matches DailyTalkCond format: data_stereo/<n>.wav + <n>.json{alignments,text_conditions} + data.jsonl."""
import os, re, json, glob, argparse, collections, tempfile
import numpy as np, sphn
from huggingface_hub import snapshot_download

SR = 24000
WIN = 90.0            # window seconds (<= duration_sec)
HOP = 75.0
# bucket thresholds from data percentiles (IEMOCAP 1-5 scale)
ARO_LO, ARO_HI = 2.67, 3.5
DOM_LO, DOM_HI = 3.0, 3.5
EMAP = {"neu":"neu","hap":"hap","exc":"hap","ang":"ang","fru":"ang","sad":"sad"}

def b3(x, lo, hi):
    return "lo" if x < lo else ("hi" if x > hi else "mid")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    ow = os.path.join(a.out, "data_stereo"); os.makedirs(ow, exist_ok=True)

    p = snapshot_download("Ar4ikov/iemocap_audio_text_splitted", repo_type="dataset",
                          cache_dir="/iopsstor/scratch/cscs/mrohania/hf_cache", allow_patterns=["*.parquet"])
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(p + "/**/*.parquet", recursive=True)):
        if "test-" in f: continue
        t = pq.read_table(f, columns=["titre","start_time","end_time","activation","dominance","emotion","to_translate","audio"])
        rows += t.to_pylist()
    # group by dialogue
    dlg = collections.defaultdict(list)
    for r in rows:
        m = re.match(r"(.+?)_([FM])(\d+)$", r["titre"] or "")
        if not m: continue
        dlg[m.group(1)].append((m.group(2), int(m.group(3)), r))
    keys = sorted(dlg.keys())
    keys = [k for i,k in enumerate(keys) if i % a.nshards == a.shard]

    def load_utt(r):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(r["audio"]["bytes"]); tp = tf.name
        try:
            d, sr = sphn.read(tp)          # [C,T] float32
        finally:
            os.unlink(tp)
        w = d.mean(0) if d.ndim > 1 else d
        if sr != SR:
            w = sphn.resample(w, sr, SR)
        return np.asarray(w, dtype=np.float32)

    n = 0
    jo = open(os.path.join(a.out, f"shard_{a.shard}.jsonl"), "w")
    for k in keys:
        utts = dlg[k]
        spk_counts = collections.Counter(s for s,_,_ in utts)
        if len(spk_counts) < 2: continue
        t0 = min(r["start_time"] for _,_,r in utts)
        end = max(r["end_time"] for _,_,r in utts)
        # generate both channel assignments (each speaker as MAIN once)
        for main_spk in spk_counts:
            ws = t0
            while ws < end:
                we = ws + WIN
                win_utts = [(s,r) for s,_,r in utts if r["start_time"] < we and r["end_time"] > ws]
                mains = [r for s,r in win_utts if s == main_spk]
                others = [r for s,r in win_utts if s != main_spk]
                if len(mains) >= 2 and len(others) >= 1:
                    buf = np.zeros((2, int(WIN*SR)+SR), dtype=np.float32)
                    aligns = []
                    emos = []; aros = []; doms = []
                    for s, r in win_utts:
                        ch = 0 if s == main_spk else 1
                        off = r["start_time"] - ws
                        try:
                            wav = load_utt(r)
                        except Exception:
                            continue
                        st = int(max(0.0, off) * SR)
                        seg = wav[max(0, int(-off*SR)):]
                        L = min(len(seg), buf.shape[1]-st)
                        if L > 0: buf[ch, st:st+L] += seg[:L]
                        # word-level alignment: distribute words evenly over [start,end]
                        txt = (r["to_translate"] or "").split()
                        s0 = max(0.0, r["start_time"]-ws); s1 = min(WIN, r["end_time"]-ws)
                        who = "SPEAKER_MAIN" if ch == 0 else "SPEAKER_OTHER"
                        if txt and s1 > s0:
                            dt = (s1-s0)/len(txt)
                            for i, wtok in enumerate(txt):
                                aligns.append([wtok, [round(s0+i*dt,3), round(s0+(i+1)*dt,3)], who])
                        if ch == 0:  # conditions from MAIN (Moshi) speaker
                            e = EMAP.get(r["emotion"])
                            if e: emos.append(e)
                            aros.append(r["activation"]); doms.append(r["dominance"])
                    if aligns and emos:
                        aligns.sort(key=lambda x: x[1][0])
                        # trim buffer to last event
                        last = max(x[1][1] for x in aligns)
                        buf = buf[:, :int((last+0.5)*SR)]
                        peak = np.abs(buf).max()
                        if peak > 0: buf = buf * (0.95/peak)
                        emo = collections.Counter(emos).most_common(1)[0][0]
                        aro = b3(float(np.mean(aros)), ARO_LO, ARO_HI)
                        dom = b3(float(np.mean(doms)), DOM_LO, DOM_HI)
                        bn = f"{a.shard}_{n}"
                        sphn.write_wav(os.path.join(ow, bn+".wav"), buf, SR)
                        json.dump({"alignments": aligns,
                                   "text_conditions": {"emotion": emo, "arousal": aro, "dominance": dom}},
                                  open(os.path.join(ow, bn+".json"), "w"))
                        jo.write(json.dumps({"path": f"data_stereo/{bn}.wav", "duration": buf.shape[1]/SR})+"\n")
                        n += 1
                ws += HOP
    jo.close()
    print(f"[{a.shard}] DONE clips={n}")

if __name__ == "__main__":
    main()
