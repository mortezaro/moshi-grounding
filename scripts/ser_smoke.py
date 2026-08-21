import json, glob, numpy as np, torch, sphn
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL = "superb/hubert-large-superb-er"
dev = "cuda" if torch.cuda.is_available() else "cpu"
fe = AutoFeatureExtractor.from_pretrained(MODEL)
model = AutoModelForAudioClassification.from_pretrained(MODEL).to(dev).eval()
id2label = model.config.id2label
print("device:", dev, "labels:", id2label)

base = "/iopsstor/scratch/cscs/mrohania/datasets/DailyTalkContiguous/data_stereo"
def turns(al):
    cur=[al[0]]; 
    for a in al[1:]:
        if a[2]==cur[-1][2]: cur.append(a)
        else: yield cur; cur=[a]
    yield cur

for wavp in sorted(glob.glob(base+"/*.wav"))[:2]:
    al = json.load(open(wavp[:-4]+".json"))["alignments"]
    print("\n==", wavp.split("/")[-1])
    for turn in list(turns(al))[:4]:
        spk=turn[0][2]; t0=turn[0][1][0]; t1=turn[-1][1][1]
        ch = 0 if spk=="SPEAKER_MAIN" else 1
        data, sr = sphn.read(wavp, start_sec=t0, duration_sec=max(t1-t0,0.2), sample_rate=16000)
        seg = data[ch]
        inp = fe(seg, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            prob = model(inp.input_values.to(dev)).logits.softmax(-1)[0].cpu()
        top=int(prob.argmax())
        words=" ".join(w[0] for w in turn)[:50]
        print(f"  {spk:13s}[{t0:5.1f}-{t1:5.1f}] {id2label[top]:4s} ({float(prob[top]):.2f})  |{words}|")
