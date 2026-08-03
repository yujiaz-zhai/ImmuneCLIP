#!/usr/bin/env bash
set -euo pipefail

ART=/root/autodl-tmp/experiments/artifacts
PY=/root/miniconda3/envs/aaai/bin/python
SEED=42

mkdir -p "$ART/logs/E1_projection_existing" "$ART/outputs/E1_projection_existing"
cd "$ART/immuneclip"

export PYTHONPATH="$ART/immuneclip:/root/workspace/usenix/scripts:${PYTHONPATH:-}"
export IMMUNECLIP_EXP_ROOT="$ART/outputs/E1_projection_existing"

COMMON=(
  --ft lora
  --objective contrastive
  --steps 300
  --eval_steps 5,10,20,30,50,100,200,300
  --batch_size 16
  --lr 1e-6
  --seed "$SEED"
  --subset 1000
  --downstream cc3m
  --cc3m_root /root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K
  --cc3m_csv /root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K/cc3m_natural_10K_no_banana_strict.csv
  --revival_threshold 0.5
)

run_one() {
  local tag="$1"
  local ckpt="$2"
  echo "[$(date)] START $tag" | tee -a "$ART/logs/E1_projection_existing/master_seed${SEED}.log"
  "$PY" run_downstream.py --ckpt "$ckpt" --tag "$tag" "${COMMON[@]}" \
    2>&1 | tee "$ART/logs/E1_projection_existing/${tag}.log"
  echo "[$(date)] DONE $tag" | tee -a "$ART/logs/E1_projection_existing/master_seed${SEED}.log"
}

run_one rn50_align_par_proj_s42 \
  /root/autodl-tmp/experiments/immuneclip_week2/defense_align_ep10/checkpoints/par_cleaned_rn50.pt
run_one rn50_align_invertune_proj_s42 \
  /root/autodl-tmp/experiments/immuneclip_week3_inv_base/results/badclip_align_invertune/checkpoints/defended_model.pt
run_one rn50_badclip_par_proj_s42 \
  /root/autodl-tmp/experiments/immuneclip/week1/checkpoints/par_cleaned_rn50.pt
