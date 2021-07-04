#!/bin/bash
#SBATCH --job-name="wandb-test"
#SBATCH --output="atles-out/wandb-test.%j.%N.out"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=96G
#SBATCH --account=wmu101
#SBATCH --no-requeue
#SBATCH -t 48:00:00

module purge
module load gpu
module load slurm

CUDA_LAUNCH_BLOCKING=1 python3 run_train.py
# python3 run_train.py
