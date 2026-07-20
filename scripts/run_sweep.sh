#!/bin/bash
# Submit with:
# sbatch scripts/run_sweep.sh

#SBATCH -J sweep_260713
#SBATCH -p high
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

echo "[INFO] Starting W&B sweep agent on $(hostname) at $(date)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

cd /home/afana01t/biohack2024

mkdir -p logs/sweep_260713 models/sweep_260713 wandb

poetry install --no-root

export PYTHONPATH=/home/afana01t/biohack2024/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=/home/afana01t/biohack2024/wandb

poetry run python scripts/sweep.py --count 27

echo "[INFO] Finished at $(date)"
