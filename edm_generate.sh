#!/bin/bash
#SBATCH -J edm_generate                      # Job name
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --nodes=1                     # Ensure that all cores are on the same machine
#SBATCH --partition=2080-galvani      # Which partition will run your job
#SBATCH --time=2-00:05                # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:2                  # Requesting type and number of GPUs
#SBATCH --mem=50G                     # Total memory pool for all cores
#SBATCH --output=/home/geiger/gwb343/minimax/logs/generate-%j.out  # STDOUT file
#SBATCH --error=/home/geiger/gwb343/minimax/logs/generate-%j.err   # STDERR file
#SBATCH --mail-type=ALL               # Type of email notification
#SBATCH --mail-user=su_yu_zheng@hotmail.com  # Email for notifications

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus
ls $WORK # not necessary just here to illustrate that $WORK is available here

source ~/.bashrc
conda activate /home/geiger/gwb343/.conda/envs/diff
torchrun --nnode=1 --nproc_per_node=1  --master_port=25680 generate.py --outdir=../edm/generated_017_0003120 \
--seeds=0-49 --batch=1 --network=../logs/cifar10/017-EDM-minimax/checkpoints/0003120.pt \
--class 0 --subdirs
conda deactivate