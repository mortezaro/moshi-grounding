#!/bin/bash
# Source this, then run python via: uenv run --view=default pytorch/v2.9.1:v2 -- $MOSHI_PY ...
# or use the moshi_py() helper below.
export SCR=/iopsstor/scratch/cscs/mrohania
export MOSHI_PY=$SCR/moshivenv/bin/python
# HF caches on scratch (home dir has a tiny quota)
export HF_HOME=$SCR/hf_home
export HF_HUB_CACHE=$SCR/hf_home/hub
export HUGGINGFACE_HUB_CACHE=$SCR/hf_home/hub
# runtime libopus (built from source) for sphn
export LD_LIBRARY_PATH=$SCR/opt/opus/lib:$LD_LIBRARY_PATH
# rust (only needed for rebuilds)
export RUSTUP_HOME=$SCR/rust/rustup CARGO_HOME=$SCR/rust/cargo
export PATH=$SCR/shims:$CARGO_HOME/bin:$PATH
# scratch tmp/pip caches
export TMPDIR=$SCR/tmp PIP_CACHE_DIR=$SCR/.pipcache
mkdir -p "$HF_HOME/hub" "$TMPDIR"
# helper: run python inside the uenv with all env applied
moshi_py() { uenv run --view=default pytorch/v2.9.1:v2 -- "$MOSHI_PY" "$@"; }
# --- torch.compile/triton fixes (GH200 inductor+home-quota) ---
export NO_TORCH_COMPILE=1
export TRITON_CACHE_DIR=$SCR/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=$SCR/.inductor_cache
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
