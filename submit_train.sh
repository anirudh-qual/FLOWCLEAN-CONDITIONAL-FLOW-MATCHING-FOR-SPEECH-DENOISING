#!/bin/bash
#SBATCH -J Flowclean_train
#SBATCH -o Report-%j.out      
#SBATCH -t 1:00:00
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:2        
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vvobbilisetty6@gatech.edu
#SBATCH -q coc-ice

cd $HOME/scratch/DL_Pro/FLOWCLEAN-CONDITIONAL-FLOW-MATCHING-FOR-SPEECH-DENOISING


# Create logs directory if it doesn't exist
mkdir -p checkpoints
source /home/hice1/vvobbilisetty6/scratch/miniconda3/etc/profile.d/conda.sh
conda activate flowclean
source .env

# ---------- Training ----------
# torchrun handles DDP; --nproc_per_node must match --ntasks-per-node above
torchrun --nproc_per_node=2 train.py --config configs/default.yaml


