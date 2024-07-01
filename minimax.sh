#!/bin/bash
#SBATCH -J minimax                      # Job name
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --nodes=1                     # Ensure that all cores are on the same machine
#SBATCH --partition=2080-galvani      # Which partition will run your job
#SBATCH --time=2-00:05                # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:2                  # Requesting type and number of GPUs
#SBATCH --mem=50G                     # Total memory pool for all cores
#SBATCH --output=/home/geiger/gwb343/minimax/logs/minimax-%j.out  # STDOUT file
#SBATCH --error=/home/geiger/gwb343/minimax/logs/minimax-%j.err   # STDERR file
#SBATCH --mail-type=ALL               # Type of email notification
#SBATCH --mail-user=su_yu_zheng@hotmail.com  # Email for notifications

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus
ls $WORK # not necessary just here to illustrate that $WORK is available here

source ~/.bashrc
conda activate /home/geiger/gwb343/.conda/envs/diff
torchrun --nnode=1 --master_port=25678 train_cond_unet2d.py --model UNet2D --dataset cifar10 \
--global-batch-size 2 --tag minimax --ckpt-every 48000 --log-every 6000 --epochs 50 \
--condense --finetune-ipc -1 --results-dir ../logs/cifar10 --size 32 --download  # Replace with your actual script or command
conda deactivate
