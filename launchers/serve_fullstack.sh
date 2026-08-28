#!/bin/bash
#SBATCH --job-name=serve_full
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
export PYTHONUNBUFFERED=1
export HF_HOME=/iopsstor/scratch/cscs/mrohania/hf_home
export COND_EMOTION=neu
export COND_DPO_LORA=/iopsstor/scratch/cscs/mrohania/dpo_runs/reg40_b05s60_E_long/dpo_lora.safetensors
C=/iopsstor/scratch/cscs/mrohania/base_models/E_long
echo "NODE=$(hostname) MODEL=E_long+emoDPO(reg40_b05s60)+greeting+reactive PORT=8907"
srun --ntasks=1 uenv run --view=default pytorch/v2.9.1:v2 -- "$MOSHI_PY" -m moshi.reactive_server --lora-weight $C/lora.safetensors --config-path $C/config.json --host 0.0.0.0 --port 8907 2>&1
