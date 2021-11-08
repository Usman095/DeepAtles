#!/bin/bash
#SBATCH --job-name="atles-hpg-atles"
#SBATCH --output="atles-out/bak/atles.%j.%N.out"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=250gb
#SBATCH --account=fsaeed
#SBATCH --no-requeue
#SBATCH -t 48:00:00

mkdir atles-out/$SLURM_JOB_ID
mkdir atles-out/$SLURM_JOB_ID/models
mkdir atles-out/$SLURM_JOB_ID/code
cp -R src config.ini read_spectra.py read_spectra.sh run_train.py train.sh atles-out/$SLURM_JOB_ID/code/

# CUDA_LAUNCH_BLOCKING=1 python3 run_train.py
python3 run_train.py