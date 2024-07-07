#!/bin/bash
#SBATCH -J edm_eval                      # Job name
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --nodes=1                     # Ensure that all cores are on the same machine
#SBATCH --partition=2080-galvani      # Which partition will run your job
#SBATCH --time=2-00:05                # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:2                  # Requesting type and number of GPUs
#SBATCH --mem=50G                     # Total memory pool for all cores
#SBATCH --output=/home/geiger/gwb343/minimax/logs/edm-eval-%j.out  # STDOUT file
#SBATCH --error=/home/geiger/gwb343/minimax/logs/edm-eval-%j.err   # STDERR file
#SBATCH --mail-type=ALL               # Type of email notification
#SBATCH --mail-user=su_yu_zheng@hotmail.com  # Email for notifications

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus
ls $WORK # not necessary just here to illustrate that $WORK is available here

source ~/.bashrc
conda activate /home/geiger/gwb343/.conda/envs/diff
python train.py -d cifar10 --val_dir ~/minimax/edm/generated_011_0011700 ./datasets \
-n convnet --nclass 10 --norm_type instance --ipc 50 --tag test --slct_type random --depth 3 --size 32 \
--batch_size 128 --lr 1e-3 --save_ckpt True --verbose --test
conda deactivate