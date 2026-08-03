#!/usr/bin/env bash
set -euo pipefail

ART=/root/autodl-tmp/experiments/artifacts
PY=/root/miniconda3/envs/aaai/bin/python
SEED=42

EXP=E2_causality_main_fix1
mkdir -p "$ART/logs/$EXP" "$ART/outputs/$EXP"
cd "$ART/immuneclip"

export PYTHONPATH="$ART/immuneclip:/root/workspace/usenix/scripts:${PYTHONPATH:-}"

COMMON=(
  --ckpt /root/autodl-tmp/experiments/immuneclip_week2/defense_align_ep10/checkpoints/par_cleaned_rn50.pt
  --score_definition trigger_margin_delta
  --steps 300
  --eval_steps 5,10,20,30,50,100,200,300
  --eval_subset 1000
  --log_every 10
  --diagnostic_every 5
  --compute_proxy_diagnostic
  --null_samples 8
  --lr 1e-6
  --batch_size 64
  --num_workers 4
  --clean_csv /root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K/cc3m_natural_10K_no_banana_strict.csv
  --cc3m_root /root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K
  --patch_path /root/autodl-tmp/experiments/immuneclip_week2/BadCLIP_GradAlign/opti_patches/ours_middle_ep10_02.jpg
  --patch_size 16
  --patch_location middle
  --proxy_trigger_path /root/autodl-tmp/experiments/immuneclip_week3_blackbox_stage0_formal_ce_rank/proxy_trigger.pt
  --target_label banana
  --param_scope selected
  --param_keywords visual.layer4 visual.attnpool
  --seed "$SEED"
)

run_one() {
  local tag="$1"
  shift
  echo "[$(date)] START $tag" | tee -a "$ART/logs/$EXP/master_seed${SEED}.log"
  "$PY" run_gradient_rebound_causality.py \
    --out_dir "$ART/outputs/$EXP/$tag" \
    --tag "$tag" \
    "$@" \
    "${COMMON[@]}" \
    2>&1 | tee "$ART/logs/$EXP/${tag}.log"
  echo "[$(date)] DONE $tag" | tee -a "$ART/logs/$EXP/master_seed${SEED}.log"
}

run_one rn50_align_par_normal_s42 \
  --mode normal
run_one rn50_align_par_project_oracle_harmful_s42 \
  --mode project --project_trigger oracle --project_harmful_only
run_one rn50_align_par_random_matched_s42 \
  --mode project --project_trigger random_matched --project_harmful_only
