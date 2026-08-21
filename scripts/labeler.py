"""Unified emotion labeler: audeering-dimensional (calibrated) or SER4 (trained classifier).
predict() returns a list of emotion strings (neu/hap/ang/sad), one per segment."""
import os, json, torch, ser_dim
_TH=json.load(open(os.path.join(os.path.dirname(__file__),"thresholds.json")))
LAB=["neu","hap","ang","sad"]  # MUST match ser_train.py order
import math
_PRIOR={"neu":166,"hap":247,"ang":345,"sad":144}  # SER4 training-class counts
_LOGPRIOR=[math.log(_PRIOR[l]/sum(_PRIOR.values())) for l in LAB]
def _emo(a,vl):
    ah=a>=_TH["aro_hi"]; al=a<=_TH["aro_lo"]; vh=vl>=_TH["val_hi"]; vll=vl<=_TH["val_lo"]
    if ah and vll: return "ang"
    if ah and vh: return "hap"
    if al and vll: return "sad"
    if vh: return "hap"
    if vll: return "sad"
    return "neu"
def load(kind, dev):
    if kind=="meld4":
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
        fe=AutoFeatureExtractor.from_pretrained("/iopsstor/scratch/cscs/mrohania/models/meld4")
        m=Wav2Vec2ForSequenceClassification.from_pretrained("/iopsstor/scratch/cscs/mrohania/models/meld4").to(dev).eval()
        return ("meld4", fe, m)
    if kind=="ser4":
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
        fe=AutoFeatureExtractor.from_pretrained("/iopsstor/scratch/cscs/mrohania/models/ser4")
        m=Wav2Vec2ForSequenceClassification.from_pretrained("/iopsstor/scratch/cscs/mrohania/models/ser4").to(dev).eval()
        return ("ser4", fe, m)
    p,m=ser_dim.load(dev); return ("audeering", p, m)
def predict(handle, dev, segs):
    kind,a,b=handle
    if not segs: return []
    if kind=="meld4":
        out=[]; B=16
        for i in range(0,len(segs),B):
            inp=a(segs[i:i+B], sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(dev)
            with torch.no_grad(): out+=[LAB[j] for j in b(inp).logits.argmax(-1).cpu().tolist()]
        return out
    if kind=="ser4":
        out=[]; B=16
        for i in range(0,len(segs),B):
            inp=a(segs[i:i+B], sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(dev)
            with torch.no_grad():
                lg=b(inp).logits.cpu()
                adj=lg - torch.tensor(_LOGPRIOR)  # logit adjustment: remove training prior
                out+=[LAB[j] for j in adj.argmax(-1).tolist()]
        return out
    vad=ser_dim.predict(a,b,dev,segs); return [_emo(x[0],x[2]) for x in vad]
