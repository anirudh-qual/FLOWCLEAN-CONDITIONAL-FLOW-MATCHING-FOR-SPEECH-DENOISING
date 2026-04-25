#!/bin/bash
#SBATCH -J fc_phase0
#SBATCH -o Report-phase0-%j.out
#SBATCH -t 24:00:00
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vvobbilisetty6@gatech.edu
#SBATCH -q coc-ice

cd $HOME/scratch/DL_Pro/FLOWCLEAN-CONDITIONAL-FLOW-MATCHING-FOR-SPEECH-DENOISING

mkdir -p checkpoints/phase0_baseline
source /home/hice1/vvobbilisetty6/scratch/miniconda3/etc/profile.d/conda.sh
conda activate flowclean
source .env

torchrun --nproc_per_node=2 train.py --config configs/default.yaml
