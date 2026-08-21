from datasets import load_dataset, Audio
from collections import Counter
def no_audio(ds):
    for name,feat in ds.features.items():
        if getattr(feat,"__class__",None).__name__=="Audio" or (hasattr(feat,"decode")):
            try: ds=ds.cast_column(name, Audio(decode=False))
            except Exception: pass
    return ds
print("===== kalbin/fisher-sft-v2 (one row, text/timing fields) =====")
try:
    ds=load_dataset("kalbin/fisher-sft-v2", split="train", streaming=True)
    ds=no_audio(ds)
    ex=next(iter(ds))
    for k,v in ex.items():
        print("  %s: %s"%(k, str(v)[:220]))
except Exception as e:
    print("  ERR", type(e).__name__, str(e)[:200])
print("===== NathanRoll/speech-emotion-dataset-english (labels over 1500) =====")
try:
    ds=load_dataset("NathanRoll/speech-emotion-dataset-english", split="train", streaming=True)
    ds=no_audio(ds)
    c=Counter(); dsrc=Counter(); n=0
    for ex in ds:
        c[ex.get("emotion")]+=1; dsrc[ex.get("dataset")]+=1; n+=1
        if n>=1500: break
    print("  emotions:", dict(c))
    print("  sources:", dict(dsrc.most_common(10)))
except Exception as e:
    print("  ERR", type(e).__name__, str(e)[:200])
