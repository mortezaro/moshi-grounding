"""Dimensional SER: audeering wav2vec2 -> (arousal, dominance, valence) in [0,1]."""
import torch, torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

MODEL="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense=nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout=nn.Dropout(config.final_dropout)
        self.out_proj=nn.Linear(config.hidden_size, config.num_labels)
    def forward(self,x):
        x=self.dropout(x); x=self.dense(x); x=torch.tanh(x); x=self.dropout(x)
        return self.out_proj(x)

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config=config
        self.wav2vec2=Wav2Vec2Model(config)
        self.classifier=RegressionHead(config)
        self.init_weights()
    def forward(self, input_values):
        h=self.wav2vec2(input_values)[0]
        h=torch.mean(h,dim=1)
        return h, self.classifier(h)   # logits: [arousal, dominance, valence]

def load(dev):
    proc=Wav2Vec2Processor.from_pretrained(MODEL)
    model=EmotionModel.from_pretrained(MODEL).to(dev).eval()
    return proc, model

@torch.no_grad()
def predict(proc, model, dev, segs, sr=16000, B=16):
    out=[]
    for i in range(0,len(segs),B):
        chunk=segs[i:i+B]
        x=proc(chunk, sampling_rate=sr, return_tensors="pt", padding=True).input_values.to(dev)
        _,logits=model(x)
        out+=logits.cpu().tolist()   # each [aro,dom,val]
    return out
