# moshi-grounding

Making **Kyutai Moshi** (full-duplex speech-to-speech) *emotion- and state-aware* — without losing its live conversational quality. A common, reproducible starting point for the grounding experiments: vendored base model + training wrapper + data/eval scripts, so everyone branches from an identical base.

## What's here

```
moshi/        vendored base Moshi (patched for conditioning — see "Patches")
finetune/     training wrapper (LoRA/FSDP): train.py + finetune/ + example configs
scripts/      data building + evaluation (build_conditioned*, ser_dim, fluency, affect/valence A/B, quality)
launchers/    SLURM sbatch scripts (Alps/clariden)
env/          moshi_env.sh — uenv + venv activation
docs/         EXPERIMENTS.md (full run ledger) · ALPS_WORKFLOW.md (cluster ops)
tools/        retarget.sh — repoint hardcoded scratch paths to yours
```

## The idea, in one paragraph

Grounding **as tokens** in the inner-monologue stream lifts offline recognition but breaks the live conversation (the model emits tags instead of speech). Grounding **as conditioning** — emotion/arousal/dominance/valence as LUT conditioner *inputs*, sum-fused into the transformer, never spoken — preserves clean speech. The validated baseline is **`S_cond_v3`** (100% DailyTalk, emotion conditioning): fluent, base voice quality preserved, emotion-aware. Making it actually *steer* affect is the open frontier; current lead is **dimensional conditioners** (not categorical emotion) + conditioner LR upweight. Full history and numbers in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Setup (Alps / clariden)

```bash
# 1. env: heavy stack from the uenv, extras in a --system-site-packages venv
source env/moshi_env.sh            # sets $MOSHI_PY, HF_HOME, LD_LIBRARY_PATH, NO_TORCH_COMPILE=1

# 2. use the VENDORED moshi (do not pip install moshi — it would shadow our patches)
export PYTHONPATH="$PWD/moshi:$PWD/finetune:$PYTHONPATH"

# 3. point scripts/launchers at your scratch (they currently hardcode mrohania's)
tools/retarget.sh /iopsstor/scratch/cscs/<youruser>
```

See [`docs/ALPS_WORKFLOW.md`](docs/ALPS_WORKFLOW.md) for SLURM/backfill/automation practices.

## Reproduce the baseline

```bash
# build conditioned data (SER labels) then train S_cond_v3-style
sbatch launchers/train_scond3_16.sh      # 16 nodes, ~3.6 h, 800 steps
# evaluate: live fluency + quality gate + affect A/B
$MOSHI_PY scripts/fluency_audio.py --adapter <ckpt>/consolidated/lora.safetensors --n 24
$MOSHI_PY scripts/affect_ab.py    --adapter <ckpt>/consolidated/lora.safetensors --n 14 --cfg 1.0
```

Datasets and checkpoints are **not** in git (see `.gitignore`); regenerate data with `scripts/build_conditioned*.py` and fetch the base model from HF (`kyutai/moshiko-pytorch-bf16`).

## Patches to base Moshi (why it's vendored)

Our conditioning work required edits to the base package, kept here as the vendored `moshi/`:
- `models/lm.py` — embed conditions **inside** `forward` (calling the conditioner separately breaks FSDP `_is_root`).
- `run_inference.py` — `get_condition_tensors` builds per-attribute condition tensors from `COND_<ATTR>` env (per-attr valid defaults: emotion→neu, aro/dom/pit→mid).
- `conditioners/base.py`, `models/loaders.py` — FSDP-safe prepare/unwrap; materialize meta conditioner params before device move; `strict=False` base load.

The training wrapper adds conditioner injection + trainable conditioners + a `COND_LR_MULT` param-group (`finetune/train.py`), gated by env flags: `COND_EMOTION`, `COND_COG` (arousal+dominance), `COND_VAL` (valence), `COND_LR_MULT`.

## Branch strategy

- `main` — the common base: vendored moshi + wrapper + validated recipe. Keep it reproducible.
- `exp/<topic>` or `<person>/<idea>` — experiments off `main` (`exp/valence`, `exp/upweight`, …). Push freely.
- Tag validated milestones (e.g. `v3-baseline`).

## Provenance

Vendors [Kyutai Moshi](https://github.com/kyutai-labs/moshi) and [moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) with local patches for grounding; upstream licenses retained in `moshi/` and `finetune/`. This repo is private and research-internal.
