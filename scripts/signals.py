"""Deterministic conversational-state signals from stereo DailyTalk audio (no labels needed).
L=SPEAKER_MAIN (Moshi), R=user. All per user-turn unless noted."""
import numpy as np

def frame_energy(x, sr, hop=0.02):
    n=max(1,int(hop*sr)); m=len(x)//n
    if m==0: return np.array([0.0])
    return np.sqrt((x[:m*n].reshape(m,n).astype(np.float32)**2).mean(1)+1e-9)

def db(x): return 20*np.log10(max(x,1e-9))

def response_latency(gap_r, sr, gap_start, prev_main_end, thr=0.02, hop=0.02):
    """User response latency: time from end of prev Moshi turn to user speech onset in the gap."""
    e=frame_energy(gap_r, sr, hop)
    on=np.argmax(e>thr) if (e>thr).any() else 0
    onset_time = gap_start + on*hop
    return max(0.0, onset_time - prev_main_end)

def proximity_level(seg_r, sr):
    """Proxy for mic proximity: speech RMS in dB (louder ~ closer). Returns dB."""
    e=frame_energy(seg_r, sr)
    speech=e[e>np.percentile(e,60)]
    return db(speech.mean() if len(speech) else e.mean())

def snr_db(seg_r, sr):
    """Ambient: SNR = speech level vs noise floor (low-energy frames)."""
    e=frame_energy(seg_r, sr)
    noise=np.percentile(e,10); speech=np.percentile(e,90)
    return db(speech)-db(noise)

def overlap_fraction(L, R, sr, thr=0.02, hop=0.02):
    """Competing speech: fraction of frames where BOTH channels are active."""
    eL=frame_energy(L,sr,hop); eR=frame_energy(R,sr,hop)
    m=min(len(eL),len(eR)); 
    both=((eL[:m]>thr)&(eR[:m]>thr)).mean()
    return float(both)

def bucketize(v, edges, labels):
    for e,l in zip(edges,labels[:-1]):
        if v<e: return l
    return labels[-1]

# ---- cognitive-state signals (from transcript + timing) ----
FILLERS={"uh","um","er","erm","hmm","uhh","umm","mm","mhm","huh"}
def speech_rate(words, t0, t1):
    """words per second over the turn."""
    dur=max(t1-t0,1e-3); return len(words)/dur
def disfluency_rate(words):
    """fraction of tokens that are fillers or immediate repetitions (hesitation proxy)."""
    if not words: return 0.0
    w=[x.lower().strip(".,?!") for x in words]
    fill=sum(1 for x in w if x in FILLERS)
    rep=sum(1 for i in range(1,len(w)) if w[i]==w[i-1] and w[i])
    return (fill+rep)/len(w)
def intra_turn_pause(alz):
    """max silent gap between consecutive words within a turn (processing/hesitation)."""
    g=[alz[i+1][1][0]-alz[i][1][1] for i in range(len(alz)-1)]
    return max(g) if g else 0.0

# ---- physical: pitch (f0) via autocorrelation ----
def pitch_hz(x, sr, fmin=70, fmax=350):
    """Median voiced f0 (Hz) of a segment via autocorrelation. 0 if unvoiced."""
    import numpy as np
    if len(x) < sr*0.05: return 0.0
    x = x.astype(np.float32); x = x - x.mean()
    hop=int(0.03*sr); win=int(0.05*sr); f0s=[]
    lo=int(sr/fmax); hi=int(sr/fmin)
    for i in range(0, max(1,len(x)-win), hop):
        f=x[i:i+win]
        if np.sqrt((f**2).mean()) < 0.01: continue
        ac=np.correlate(f,f,"full")[len(f)-1:]
        if len(ac)<=hi: continue
        seg=ac[lo:hi]
        if len(seg)==0 or ac[0]<=0: continue
        lag=lo+int(np.argmax(seg))
        if ac[lag] > 0.3*ac[0]: f0s.append(sr/lag)
    return float(np.median(f0s)) if f0s else 0.0
