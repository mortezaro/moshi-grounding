# Running the emotion-aware Moshi models (anywhere)

This guide takes you from a fresh machine with a GPU to a **live, talkable** emotion-aware
Moshi — independent of clariden. Everything you need is in this repo; the only external
download is the public base model from Hugging Face.

## The 4 models we keep

All are **LoRA adapters** on top of the public base **`kyutai/moshiko-pytorch-bf16`** (~8 GB,
downloads itself from HF). The adapters live in [`models/`](../models).

| Name | Path | What it is |
|---|---|---|
| **E_long** | `models/E_long/lora.safetensors` | Expresso-emotion fine-tune, quality-first (best-behaved base) |
| **E_u8** | `models/E_u8/lora.safetensors` | Expresso-emotion fine-tune, balanced |
| **E_u12** | `models/E_u12/lora.safetensors` | Expresso-emotion fine-tune, more expressive |
| **Companion** | `models/companion_E_long/dpo_lora.safetensors` | Emotion-appropriateness **DPO adapter** — stacks on **E_long** (this is the one that passed the live test) |

The **companion** = E_long base adapter **+** the companion DPO adapter **+** greeting-prime **+**
live reactive emotion (see "Full companion" below).

## 1. What you need

- **A GPU with ≥ 16 GB VRAM** (base Moshi is ~7B in bf16). Any of: a local NVIDIA GPU, or a
  rented cloud GPU (see "Where to run it" at the end). A single A10/A100/L4/4090/H100/GH200 is plenty.
- **Python 3.10+**, PyTorch with CUDA, and this repo.
- ~10 GB disk for the base model download.

## 2. Setup

```bash
git clone https://github.com/mortezaro/moshi-grounding.git
cd moshi-grounding

# python env
python -m venv .venv && source .venv/bin/activate
pip install torch                      # the CUDA build matching your GPU/driver
pip install -U moshi sphn numpy safetensors sentencepiece huggingface_hub

# IMPORTANT: use the VENDORED, patched moshi in this repo (not the pip one) — it has our
# conditioning + reactive patches. Put it first on the path:
export PYTHONPATH="$PWD/moshi:$PWD/finetune:$PYTHONPATH"
```

The first server launch downloads `kyutai/moshiko-pytorch-bf16` from HF automatically (cached
in `~/.cache/huggingface` — set `HF_HOME` to move it).

## 3. Serve — three ways

### (a) Basic emotion-aware chat — any base model

`moshi.server` gives you the web mic UI. Serve a base fine-tune (emotion is fixed per launch via
`COND_EMOTION` = `neu` / `hap` / `sad`):

```bash
export COND_EMOTION=neu
python -m moshi.server \
  --lora-weight models/E_long/lora.safetensors \
  --config-path models/E_long/config.json \
  --host 0.0.0.0 --port 8998
```

Open `http://localhost:8998`, allow the mic, connect, talk.

### (b) The full companion — the one you liked ⭐

This stacks the **companion DPO adapter** on E_long, adds a **greeting** at each connection, and
**reacts live** to your emotion (an SER reads your voice and shifts Moshi's tone ~every 1.5 s).
It uses `moshi.reactive_server` (in this repo) plus a one-line patch to `lm.py`.

```bash
# one-time: install the reactive server + greeting hook into your moshi package
cp scripts/reactive_server.py "$(python -c 'import moshi,os;print(os.path.dirname(moshi.__file__))')/reactive_server.py"
python scripts/apply_patches.py     # applies the on_text_hook return patch to lm.py (idempotent)

# run it
export COND_EMOTION=neu
export COND_DPO_LORA="$PWD/models/companion_E_long/dpo_lora.safetensors"   # stacks the emotion-DPO
python -m moshi.reactive_server \
  --lora-weight models/E_long/lora.safetensors \
  --config-path models/E_long/config.json \
  --host 0.0.0.0 --port 8907
```

You'll see in the log: `loaded emotion-DPO LoRA: …`, `reactive SER loaded`, and
`greeting-prime installed` on each connection. Open `http://localhost:8907`, connect, talk — it
opens with a greeting and warms/gentles to match how you sound.

Requires `ser_dim.py` importable (it's in `scripts/`; add `scripts/` to `PYTHONPATH`) and
`transformers` if you also want the offline judge.

### (c) Text-generation only (no mic, for quick checks)

```bash
python scripts/gen_dpo.py --adapter models/E_long/lora.safetensors \
  --config models/E_long/config.json \
  --dpo models/companion_E_long/dpo_lora.safetensors --label companion
```

## 4. Reaching a remote server from your browser

If the GPU is on a remote machine, forward the port to your laptop (mic needs a secure/local
origin):

```bash
ssh -L 8907:<gpu-host-or-node>:8907 <remote>
# then open http://localhost:8907
```

To share a public link (from a machine with open internet — e.g. your laptop, not a locked-down
cluster), run a tunnel over the forwarded port:

```bash
brew install cloudflared          # or the linux binary
cloudflared tunnel --url http://localhost:8907
```

It prints an `https://…trycloudflare.com` URL (HTTPS → remote mics work). The link lives only
while the tunnel + server run.

## 5. Where to run it once clariden is gone

You need a GPU. Options, cheapest-effort first:

- **Rented cloud GPU by the hour** — [RunPod](https://runpod.io), [Vast.ai](https://vast.ai),
  [Lambda](https://lambdalabs.com). Spin up an L4/A10/A100 (~$0.4–2/hr), `git clone` this repo,
  follow §2–3. Best for on-demand demos; you only pay while it runs.
- **A persistent cloud VM with a GPU** (AWS `g5`, GCP `g2`, Azure `NC`) — same steps, always-on,
  more expensive. Put `cloudflared` on it for a stable public link.
- **A local NVIDIA workstation** (≥16 GB VRAM) — zero ongoing cost, run §2–3 directly.
- **Hugging Face** — optional home for the *weights* (mirror `models/` to an HF model repo so
  they're loadable via `from_pretrained` / `hf://`); inference still needs a GPU (HF Inference
  Endpoints or a Space with a GPU tier).

The recipe is identical everywhere: **clone repo → install moshi → download base moshiko from HF
→ point the server at the adapters in `models/`.** Nothing depends on clariden.

## 6. If you'd rather host the weights on Hugging Face too

```bash
pip install huggingface_hub
huggingface-cli login                       # paste a write token from hf.co/settings/tokens
huggingface-cli upload mortezaro/moshi-emotion-companion models/ .
```

Then anyone can `huggingface-cli download mortezaro/moshi-emotion-companion` to get the adapters,
and you have a second permanent copy independent of GitHub.
