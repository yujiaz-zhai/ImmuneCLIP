#!/usr/bin/env bash
set -euo pipefail

ART=/root/autodl-tmp/experiments/artifacts
PY=/root/miniconda3/envs/aaai/bin/python
EXP=E3_align_par_rescue_seed42
SEED=42

INIT_CKPT=/root/autodl-tmp/experiments/immuneclip_week2/defense_align_ep10/checkpoints/par_cleaned_rn50.pt
REF_CKPT=${INIT_CKPT}
CLEAN_CSV=/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K/cc3m_natural_10K_no_banana_strict.csv
CC3M_ROOT=/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K
EVAL_STEPS=5,10,20,30,50,100,200,300

mkdir -p "${ART}/logs/${EXP}" "${ART}/outputs/${EXP}/runs" "${ART}/outputs/${EXP}/downstream"
cd "${ART}/immuneclip"

wait_for_gpu() {
  while true; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    if [[ -z "${used}" || "${used}" -lt 2500 ]]; then
      break
    fi
    echo "[$(date)] GPU busy (${used} MiB used); waiting before ${1}" | tee -a "${ART}/logs/${EXP}/master_seed${SEED}.log"
    sleep 120
  done
}

run_downstream() {
  local ckpt="$1"
  local tag="$2"
  local out_root="${ART}/outputs/${EXP}/downstream/${tag}"
  local summary="${out_root}/results/traj_${tag}_contrastive_full_s${SEED}_summary.json"
  if [[ -s "${summary}" ]]; then
    echo "[$(date)] SKIP downstream ${tag}" | tee -a "${ART}/logs/${EXP}/master_seed${SEED}.log"
    return
  fi
  mkdir -p "${out_root}"
  echo "[$(date)] START downstream ${tag}" | tee -a "${ART}/logs/${EXP}/master_seed${SEED}.log"
  IMMUNECLIP_EXP_ROOT="${out_root}" "${PY}" run_downstream.py \
    --ckpt "${ckpt}" \
    --ft full \
    --objective contrastive \
    --steps 300 \
    --eval_steps "${EVAL_STEPS}" \
    --batch_size 16 \
    --lr 1e-6 \
    --seed "${SEED}" \
    --tag "${tag}" \
    --subset 1000 \
    --downstream cc3m \
    --cc3m_root "${CC3M_ROOT}" \
    --cc3m_csv "${CLEAN_CSV}" \
    --revival_threshold 0.5 \
    --target_label banana \
    --patch_type ours_tnature \
    --patch_location middle \
    --patch_size 16 \
    --patch_name opti_patches/badCLIP.jpg \
    2>&1 | tee "${ART}/logs/${EXP}/downstream_${tag}.log"
  echo "[$(date)] DONE downstream ${tag}" | tee -a "${ART}/logs/${EXP}/master_seed${SEED}.log"
}

run_train() {
  local tag="$1"
  shift
  local out="${ART}/outputs/${EXP}/runs/${tag}"
  local ckpt="${out}/checkpoints/${tag}_final.pt"
  if [[ -s "${ckpt}" ]]; then
    echo "[$(date)] SKIP train ${tag}" | tee -a "${ART}/logs/${EXP}/master_seed${SEED}.log"
  else
    wait_for_gpu "${tag}"
    mkdir -p "${out}/logs"
    echo "[$(date)] START train ${tag}" | tee -a "${ART}/logs/${EXP}/master_seed${SEED}.log"
    "${PY}" immuneclip_new_train.py \
      --init_ckpt "${INIT_CKPT}" \
      --ref_ckpt "${REF_CKPT}" \
      --clean_csv "${CLEAN_CSV}" \
      --cc3m_root "${CC3M_ROOT}" \
      --steps 80 \
      --batch_size 16 \
      --update_batch_size 16 \
      --num_workers 0 \
      --train_scope selected \
      --param_keywords visual.attnpool \
      --proxy_variants 4 \
      --proxies_per_step 0 \
      --num_update_dirs 4 \
      --update_dir_modes grad sign_precond \
      --lambda_clip 1.0 \
      --lambda_kd 5.0 \
      --lambda_anchor 0.05 \
      --lambda_dir 0.15 \
      --lambda_reach 0.15 \
      --lr 5e-7 \
      --weight_decay 0.01 \
      --max_grad_norm 0.05 \
      --reach_radius 8e-5 \
      --dir_smooth cvar \
      --reach_smooth cvar \
      --log_every 1 \
      --save_every 40 \
      --eval_every 40 \
      --eval_subset 1000 \
      --seed "${SEED}" \
      --out_dir "${out}" \
      --tag "${tag}" \
      "$@" \
      2>&1 | tee "${ART}/logs/${EXP}/train_${tag}.log"
    echo "[$(date)] DONE train ${tag}" | tee -a "${ART}/logs/${EXP}/master_seed${SEED}.log"
  fi
  wait_for_gpu "downstream ${tag}"
  run_downstream "${ckpt}" "${tag}_rebound_full"
}

echo "[$(date)] E3 rescue seed=${SEED} queued" | tee -a "${ART}/logs/${EXP}/master_seed${SEED}.log"

run_train rn50_align_par_immuneclip_traj_global_working_s42 \
  --reach_steps 1 \
  --reach_mode traj_global

echo "[$(date)] E3 rescue seed=${SEED} complete" | tee -a "${ART}/logs/${EXP}/master_seed${SEED}.log"
