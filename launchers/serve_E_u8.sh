#!/bin/bash
#SBATCH --job-name=serve_E_u8
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=11:59:00
#SBATCH --output=/iopsstor/scratch/cscs/mrohania/logs/%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/mrohania/logs/%x_%j.err
source /iopsstor/scratch/cscs/mrohania/moshi_env.sh
cd /iopsstor/scratch/cscs/mrohania/moshi_emo
export PYTHONPATH=/iopsstor/scratch/cscs/mrohania/moshi-finetune:$PYTHONPATH
export COND_EMOTION=neu
C=/iopsstor/scratch/cscs/mrohania/moshi_runs/E_u8/checkpoints/checkpoint_000800/consolidated
echo "NODE=$(hostname) MODEL=E_u8 PORT=8901"
srun --ntasks=1 uenv run --view=default pytorch/v2.9.1:v2 -- "$MOSHI_PY" -m moshi.server --lora-weight $C/lora.safetensors --config-path $C/config.json --host 0.0.0.0 --port 8901 2>&1
