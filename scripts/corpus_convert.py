#!/usr/bin/env python
"""Convert naturalistic dyadic corpora -> moshi-finetune format, so the SAME grounding
pipeline (augment_grounding.py) then runs on them.

TARGET CONTRACT (what every adapter must emit), identical to DailyTalk:
  <out>/data_stereo/<id>.wav   stereo: LEFT = main speaker (Moshi voice), RIGHT = other/user
  <out>/data_stereo/<id>.json  {"alignments": [[word,[t0,t1],"SPEAKER_MAIN"], ...]}  (main speaker only)
  <out>/<name>.jsonl           lines {"path":"data_stereo/<id>.wav","duration":<sec>}
Then: augment_grounding.py --src <out> --out <out>Ground   (adds grounding tokens)

STATUS: Fisher & CANDOR AUDIO ARE ABSENT on scratch (bit-rot). These adapters encode the
known public layouts and are UNVERIFIED until audio is present. Fill/validate on a sample.
"""
import os, json, glob, argparse

def write_dataset(out, items):
    """items: list of (id, stereo_wav_path_or_writer, alignments, duration)."""
    os.makedirs(os.path.join(out,"data_stereo"), exist_ok=True)
    with open(os.path.join(out,"data.jsonl"),"w") as jo:
        for _id, wav, al, dur in items:
            json.dump({"alignments": al}, open(os.path.join(out,"data_stereo",f"{_id}.json"),"w"))
            jo.write(json.dumps({"path":f"data_stereo/{_id}.wav","duration":dur})+"\n")

# --- CANDOR adapter (public format: per-convo audio + transcription/*.csv w/ turn start,stop,speaker,utterance)
def candor_adapter(src, out):
    """CANDOR: 2 participants. Pick one as SPEAKER_MAIN; build stereo (main L / other R);
    word timings come from the CSV (turn-level -> distribute words across [start,stop]).
    Needs: per-convo audio (missing on scratch). See https://betterup-data.github.io/candor ."""
    raise NotImplementedError("CANDOR audio absent on scratch; wire once audio is downloaded.")

# --- Fisher adapter (LDC): 2-channel .sph telephone + .txt transcript with per-utterance times/speaker
def fisher_adapter(src, out):
    """Fisher is ALREADY 2-channel (A/B) -> map A=SPEAKER_MAIN(L), B=other(R) directly.
    Transcript lines: <t0> <t1> <A|B>: text  -> distribute words over [t0,t1].
    Needs: LDC2004S13/2004T19 + 2005S13/2005T19 (missing on scratch)."""
    raise NotImplementedError("Fisher audio/transcripts absent on scratch; needs LDC license.")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("corpus", choices=["candor","fisher"])
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    a=ap.parse_args()
    {"candor":candor_adapter,"fisher":fisher_adapter}[a.corpus](a.src,a.out)
