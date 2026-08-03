#!/usr/bin/env bash
set -euo pipefail

ART=/root/autodl-tmp/experiments/artifacts
PY=/root/miniconda3/envs/aaai/bin/python
SEED=42
EXP=E2_causality_allparam_fix2
TAG=rn50_align_par_proxy_shuffled_allparam_s42

mkdir -p "$ART/logs/$EXP" "$ART/outputs/$EXP"
cd "$ART/immuneclip"

export PYTHONPATH="$ART/immuneclip:/root/workspace/usenix/scripts:${PYTHONPATH:-}"

if [[ -f "$ART/outputs/$EXP/$TAG/summary.json" ]]; then
  echo "[$(date)] SKIP $TAG summary exists" | tee -a "$ART/logs/$EXP/master_seed${SEED}.log"
  exit 0
fi

echo "[$(date)] START $TAG" | tee -a "$ART/logs/$EXP/master_seed${SEED}.log"
"$PY" run_gradient_rebound_causality.py \
  --out_dir "$ART/outputs/$EXP/$TAG" \
  --tag "$TAG" \
  --mode project \
  --project_trigger proxy_shuffled \
  --project_harmful_only \
  --ckpt /root/autodl-tmp/experiments/immuneclip_week2/defense_align_ep10/checkpoints/par_cleaned_rn50.pt \
  --score_definition trigger_margin_delta \
  --steps 300 \
  --eval_steps 5,10,20,30,50,100,200,300 \
  --eval_subset 1000 \
  --log_every 10 \
  --diagnostic_every 10 \
  --lr 1e-6 \
  --weight_decay 0.01 \
  --max_grad_norm 0.0 \
  --batch_size 64 \
  --num_workers 4 \
  --clean_csv /root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K/cc3m_natural_10K_no_banana_strict.csv \
  --cc3m_root /root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K \
  --patch_path /root/autodl-tmp/experiments/immuneclip_week2/BadCLIP_GradAlign/opti_patches/badCLIP.jpg \
  --patch_size 16 \
  --patch_location middle \
  --proxy_trigger_path /root/autodl-tmp/experiments/immuneclip_week3_blackbox_stage0_formal_ce_rank/proxy_trigger.pt \
  --target_label banana \
  --param_scope all \
  --seed "$SEED" \
  2>&1 | tee "$ART/logs/$EXP/${TAG}.log"
echo "[$(date)] DONE $TAG" | tee -a "$ART/logs/$EXP/master_seed${SEED}.log"
