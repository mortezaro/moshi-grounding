# Emotion-reactive Moshi (live)

Beyond serving Moshi *in* a fixed emotion (`COND_EMOTION=hap`), the reactive server
makes Moshi **respond to the user's emotion**: it listens to the user's voice, reads
their affect with an SER, and swaps its own emotion conditioning live, mid-conversation.

The loop is: **perceive** (SER on user audio) → **decide** (policy) → **condition**
(swap `condition_sum`) → **respond** (Moshi generates under the new emotion). Nothing is
spoken or tagged — emotion stays in the conditioner, so base Moshi quality is preserved
(the standing constraint).

## What actually works: valence, on natural speech

We validated the perception step honestly before trusting it (`scripts/nat_perceive.py`,
run on **IEMOCAP improvised** — real human emotional conversation, not acted).

SER VAD by true emotion (natural clips):

| true emotion | arousal | valence | reads as |
|---|---|---|---|
| happy   | 0.53 | **0.57** (highest) | high valence ✓ |
| sad     | 0.33 | **0.31** (lowest)  | low valence + low arousal ✓ |
| angry   | 0.58 (highest) | 0.45 | high arousal, mid valence |
| neutral | 0.48 | 0.49 | middle |

Two findings drive the design:

1. **The SER works on *natural* speech, not acted speech.** On acted RAVDESS, "happy"
   collapsed to valence 0.31 (misread as sad/angry). On natural IEMOCAP, happy correctly
   reads high valence (0.57). Live users produce natural speech → perception is reliable
   where it matters.
2. **Valence is the trustworthy axis.** It cleanly separates happy (70% correct) from sad
   (70% correct). Arousal-driven anger/neutral is muddy (4-way accuracy only ~46%). This
   matches what the *conditioner* steers best too — so we build a **valence-based** policy
   rather than fragile 4-way emotion detection.

## Policy

```
user valence (EMA-smoothed)      Moshi responds
  > 0.55  (user positive)   -->   hap   (warm, mirror)
  < 0.40  (user down/sad)   -->   sad   (gentle, empathetic)
  else                      -->   neu   (steady)
```

Empathetic-congruent: mirror positive affect, gentle with low affect, steady otherwise.
Thresholds are calibrated to the SER's natural-speech valence range (see the table).

## How the live swap works (`scripts/reactive_server.py`)

A patched `moshi.server`. In `recv_loop`, the user's raw PCM is already available before
Mimi encoding — we tap it:

- **buffer**: keep the last ~3 s of user PCM (`_react_buffer`).
- **perceive**: every ~1.5 s, if the user isn't silent, resample 24k→16k and run the
  audeering SER → valence, EMA-smoothed (`_update_emotion`).
- **swap**: on a valence-state change, recompute the conditioner sum for the new emotion
  and assign it to the *running* generation state —
  `lm_gen._streaming_state.condition_sum = fuser.get_sum(get_condition_tensors(...))`.
  Validated: neu→hap changes `condition_sum` (mean|Δ|≈0.019, shape (1,1,4096)); the
  assignment takes effect on the next generated frame with no stream reset.

Everything else (websocket, Opus, Mimi, full-duplex streaming) is unchanged, so the model
stays real-time full-duplex. The SER forward (~3 s of audio, ~100 ms on GPU) runs inline
every ~1.5 s; with RTF headroom (~0.4) this is fine for a prototype. To eliminate even a
brief hiccup, move the SER call to a worker thread (future work).

## Serve it

Installed as `moshi.reactive_server` (relative imports need the package). Launcher:
`launchers/serve_reactive.sh` — same shape as `serve_E_u8.sh` but `-m moshi.reactive_server`,
built on `L_long_best` (our best base), port 8905:

```bash
sbatch launchers/serve_reactive.sh          # 1 GPU, 12 h; node printed in the log
# reach it from the laptop exactly like docs/SERVING.md:
ssh -L 8905:<node>:8905 clariden
#   open http://localhost:8905 , allow mic, connect, and talk.
# the server log prints:  REACTIVE: user valence 0.61 -> Moshi emotion 'hap'
```

The reactive server loads both Moshi and the SER on one GPU (fits a GH200 easily).

## Status

- [x] Perception validated on natural speech (valence axis, `nat_perceive.py`).
- [x] Offline loop proven (`reactive_proto.py`): sad user → gentle reply, etc.
- [x] Live swap API validated (`_streaming_state.condition_sum` reassign works).
- [x] Live reactive server built + launched (`reactive_server.py`, `serve_reactive.sh`).
- [ ] Async SER worker thread (remove the ~100 ms/1.5 s inline blocking).
- [ ] Hysteresis tuning from live sessions (dwell time before switching).
