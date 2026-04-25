#!/bin/bash
# Sweep (solver, NFE) on a single trained checkpoint.
# Reports PESQ / ESTOI / SI-SDR per (solver, NFE) so you can plot the
# quality-vs-compute Pareto curve.
#
# Usage:
#   bash scripts/sweep_solver_nfe.sh <CHECKPOINT_PATH> <CONFIG_YAML> <OUTPUT_ROOT>
#
# Example:
#   bash scripts/sweep_solver_nfe.sh \
#     checkpoints/phase0_baseline/flowclean_best.pt \
#     configs/default.yaml \
#     enhanced/phase0_baseline

set -euo pipefail

CKPT="${1:?checkpoint path required}"
CFG="${2:-configs/default.yaml}"
OUT_ROOT="${3:-enhanced/sweep}"

# NFE values: Heun does 2 function evaluations per step, so a Heun-K run
# is comparable in compute to an Euler-(2K) run.
EULER_STEPS=(1 2 4 8 16 30)
HEUN_STEPS=(1 2 4 8 15)

mkdir -p "${OUT_ROOT}"

for K in "${EULER_STEPS[@]}"; do
  OUT_DIR="${OUT_ROOT}/euler_K${K}"
  echo "=== Euler  K=${K} -> ${OUT_DIR} ==="
  python inference.py \
    --checkpoint "${CKPT}" \
    --config "${CFG}" \
    --output_dir "${OUT_DIR}" \
    --ode_steps "${K}" \
    --solver euler \
    --eval_metrics
done

for K in "${HEUN_STEPS[@]}"; do
  OUT_DIR="${OUT_ROOT}/heun_K${K}"
  echo "=== Heun   K=${K} -> ${OUT_DIR} ==="
  python inference.py \
    --checkpoint "${CKPT}" \
    --config "${CFG}" \
    --output_dir "${OUT_DIR}" \
    --ode_steps "${K}" \
    --solver heun \
    --eval_metrics
done

echo "Done. Per-config metrics printed above."
