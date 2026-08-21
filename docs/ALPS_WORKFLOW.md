# ALPS / clariden — working notes

Alps GH200 (aarch64), whole-node `gpu:4`. Account `a0125`. All work under `/iopsstor/scratch/cscs/$USER`.

## 1. Connect
- `ssh clariden` with `-o ServerAliveInterval=15` — the link drops often (exit 255); **jobs are detached and survive** the drop, so just reconnect.
- Prefer **short commands**; for multi-line scripts, write locally and `scp` — heredocs get truncated mid-transfer on a flaky link.
- `/tmp` is **node-local** — a file scp'd to the login node is invisible on compute nodes. Put shared scripts/data on scratch.

## 2. Environment
- One activation script (`moshi_env.sh`) sets: venv python (`$MOSHI_PY`), `HF_HOME` on scratch, `LD_LIBRARY_PATH` for libopus, `NO_TORCH_COMPILE=1`, triton/inductor cache dirs.
- Always run through uenv: `uenv run --view=default pytorch/v2.9.1:v2 -- "$MOSHI_PY" script.py`.
- Gotchas: **`torchaudio.load/save` needs TorchCodec (absent) → use `sphn`** (`sphn.read/write_wav/resample`); `soundfile` also absent.

## 3. SLURM
- Partitions: `normal` (12h), `debug` (1.5h, **MaxNodes=4, 90 node-min/job cap** → ~20 steps only; useless for real runs), `low` (24h).
- Submit: `sbatch --parsable job.sh`; watch: `squeue -u $USER -h -o "%.10i %.14j %.2t %.6M %.20S %R"`.
- **Fair-share priority**: heavy usage drops it to the floor; jobs then wait behind everyone even with idle nodes (`Reason=Priority`, `StartTime` set).
- **Don't churn resubmits** — each `sbatch`/`scancel` resets the job's queue AGE, one of the few things that rebuilds priority. Submit once, leave it.
- **Backfill favors small+short jobs**: a 1-node or short-walltime job slips into gaps a 16-node/4h job can't. Size jobs to the queue, not just throughput.

## 4. Job shapes that work here
- **Training**: 16 nodes is the sweet spot — comms/data-bound, so 32 nodes is *slower*, not faster. torchrun launcher (`--nnodes=$SLURM_NNODES --nproc-per-node=4 --node-rank=$SLURM_PROCID`).
- **Eval / labeling**: 1–4 GPU, backfillable. Parallelize a sweep across the 4 GPUs of one node: `for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i ... & done; wait`.
- Expected wall time: ~3.6h for an 800-step 16-node LoRA run (≈230 GPU-h). Set `--time` with ~50% margin but no more (shorter = more backfillable).

## 5. Automation (survives disconnects)
- **Detached watchers**: `nohup bash watch.sh >/dev/null 2>&1 &` — polls on the cluster, logs to a scratch file, keeps going after SSH dies.
- **Auto-launch chains**: `until [ -f data.jsonl ]; do sleep 60; done; sbatch train.sh` — chain data-build → train → eval without babysitting.
- **Eval-on-finish**: watch for `checkpoints/checkpoint_XXXX/consolidated/lora.safetensors`, then fire **one** consolidated eval job. Avoid per-checkpoint floods — they clog your own low-priority queue.
- From the laptop side: a background poll loop (`until squeue -j $J empty; sleep`) then read the result file; don't block on it.

## 6. Systematic results
- Every stage appends to plain result files on scratch (`moshi_exp/*.txt`) and a running ledger (`EXPERIMENTS.md`) — metric + config + verdict, one row per run. Re-measure the baseline in the same job as the model for a fair delta.
- Watch eval variance: small-n generative metrics are noisy (use n≥24; re-measure `base` alongside).

## 7. Recurring gotchas
- Config parser: multi-source `train_data` must be `pathA:1.0,pathB:2.0` (no spaces) in **block** YAML, not inline `{…}` (comma breaks the flow mapping).
- Redirecting eval `stderr` to `/dev/null` hides real crashes — keep an err file when debugging "silent" skips.
- Conditioning models: their `lm.forward` asserts `fuser`/`condition_tensors` present — pass condition tensors in eval or it crashes with a blank message.
