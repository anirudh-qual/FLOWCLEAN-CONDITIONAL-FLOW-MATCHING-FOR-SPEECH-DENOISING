#!/bin/bash
#SBATCH -JFlowclean_train
#SBATCH -oReport-%j.out      
#SBATCH -t 180
#SBATCH -N1 -gres=gpu:A40:4        
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vvobbilisetty6@gatech.edu
#SBATCH -q coc-ice

cd $HOME/scratch/vvobbilisetty6/flow_clean/FLOWCLEAN-CONDITIONAL-FLOW-MATCHING-FOR-SPEECH-DENOISING


# Create logs directory if it doesn't exist
mkdir -p checkpoints
module load anaconda3
conda activate flowclean
source .env

# ---------- Training ----------
# torchrun handles DDP; --nproc_per_node must match --ntasks-per-node above
torchrun --nproc_per_node=4 train.py --config configs/default.yaml


