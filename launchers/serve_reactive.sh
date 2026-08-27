#!/bin/bash
#SBATCH --job-name=serve_reactive
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
C=/iopsstor/scratch/cscs/mrohania/base_models/L_long_best
echo "NODE=$(hostname) MODEL=L_long_reactive PORT=8905"
srun --ntasks=1 uenv run --view=default pytorch/v2.9.1:v2 -- "$MOSHI_PY" -m moshi.reactive_server --lora-weight $C/lora.safetensors --config-path $C/config.json --host 0.0.0.0 --port 8905 2>&1
