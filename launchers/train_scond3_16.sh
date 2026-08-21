#!/bin/bash
#SBATCH --job-name=moshi_Scond3
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/mrohania/logs/%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/mrohania/logs/%x_%j.err
source /iopsstor/scratch/cscs/mrohania/moshi_env.sh; export COND_EMOTION=1
cd /iopsstor/scratch/cscs/mrohania/moshi-finetune
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29665
echo "MASTER_ADDR=$MASTER_ADDR nodes=$SLURM_NNODES fluency-first no-upweight"
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "
  source /iopsstor/scratch/cscs/mrohania/moshi_env.sh; export COND_EMOTION=1
  uenv run --view=default pytorch/v2.9.1:v2 -- \"\$MOSHI_PY\" -m torch.distributed.run \
    --nnodes=$SLURM_NNODES --nproc-per-node=4 --node-rank=\$SLURM_PROCID \
    --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT -m train example/scond3.yaml"
