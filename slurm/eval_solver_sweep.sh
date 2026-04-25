#!/bin/bash
#SBATCH -J fc_sweep
#SBATCH -o Report-sweep-%j.out
#SBATCH -t 4:00:00
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vvobbilisetty6@gatech.edu
#SBATCH -q coc-ice
#
# Sweep (Euler vs Heun) x NFE on a single trained checkpoint.
# Pass CKPT and CFG via --export, e.g.:
#   sbatch --export=ALL,CKPT=checkpoints/phase0_baseline/flowclean_best.pt,CFG=configs/default.yaml,OUT=enhanced/phase0_sweep slurm/eval_solver_sweep.sh

cd $HOME/scratch/DL_Pro/FLOWCLEAN-CONDITIONAL-FLOW-MATCHING-FOR-SPEECH-DENOISING

source /home/hice1/vvobbilisetty6/scratch/miniconda3/etc/profile.d/conda.sh
conda activate flowclean
source .env

CKPT="${CKPT:-checkpoints/phase0_baseline/flowclean_best.pt}"
CFG="${CFG:-configs/default.yaml}"
OUT="${OUT:-enhanced/sweep}"

bash scripts/sweep_solver_nfe.sh "$CKPT" "$CFG" "$OUT"
