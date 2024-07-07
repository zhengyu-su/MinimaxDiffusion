#!/bin/bash
#SBATCH -J edm_train                  # Job name
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --nodes=1                     # Ensure that all cores are on the same machine
#SBATCH --partition=2080-galvani      # Which partition will run your job
#SBATCH --time=2-00:05                # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:2                  # Requesting type and number of GPUs
#SBATCH --mem=50G                     # Total memory pool for all cores
#SBATCH --output=/home/geiger/gwb343/minimax/logs/train-%j.out  # STDOUT file
#SBATCH --error=/home/geiger/gwb343/minimax/logs/train-%j.err   # STDERR file
#SBATCH --mail-type=ALL               # Type of email notification
#SBATCH --mail-user=su_yu_zheng@hotmail.com  # Email for notifications

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus
ls $WORK # not necessary just here to illustrate that $WORK is available here

source ~/.bashrc
conda activate /home/geiger/gwb343/.conda/envs/diff
torchrun --nnode=1 --nproc_per_node=2  --master_port=25679 train_edm.py --model EDM --dataset cifar10 \
--global-batch-size 128 --tag minimax --ckpt-every 3120 --log-every 390 --epochs 8 \
--condense --finetune-ipc -1 --results-dir ../logs/cifar10 --size 32 \
--pkl cifar_ckpt/edm-cifar10-32x32-cond-ve.pkl --lambda-pos 0.002 --lambda-neg 0.008 --memory-size 128 --lr 1e-4
conda deactivate