#!/usr/bin/env bash
set -euo pipefail

cd /root/workspace/usenix
source /root/miniconda3/etc/profile.d/conda.sh
conda activate aaai

EXP_ROOT=/root/autodl-tmp/experiments/immuneclip-new
RUN_ROOT=${EXP_ROOT}/runs
BATCH_LOG_DIR=${EXP_ROOT}/logs/single_proxy_formal
mkdir -p "${BATCH_LOG_DIR}" "${EXP_ROOT}/results"

INIT_CKPT=/root/autodl-tmp/experiments/immuneclip_week2/defense_align_ep10/checkpoints/par_cleaned_rn50.pt
REF_CKPT=${INIT_CKPT}
CLEAN_CSV=/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K/cc3m_natural_10K_no_banana_strict.csv
CC3M_ROOT=/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K
EVAL_STEPS=5,10,20,30,50,100,300
SEEDS=${SEEDS:-"42 43 44"}

train_common=(
  --init_ckpt "${INIT_CKPT}"
  --ref_ckpt "${REF_CKPT}"
  --clean_csv "${CLEAN_CSV}"
  --cc3m_root "${CC3M_ROOT}"
  --steps 80
  --batch_size 16
  --update_batch_size 16
  --num_workers 0
  --train_scope selected
  --param_keywords visual.attnpool
  --proxy_variants 4
  --proxies_per_step 0
  --num_update_dirs 4
  --update_dir_modes grad sign_precond
  --reach_radius 8e-5
  --lambda_clip 1.0
  --lambda_kd 5.0
  --lambda_anchor 0.05
  --lambda_dir 0.15
  --lr 5e-7
  --max_grad_norm 0.05
  --log_every 1
  --save_every 40
  --eval_every 40
  --eval_subset 1000
)

run_downstream_for_ckpt() {
  local ckpt="$1"
  local tag="$2"
  local seed="$3"
  local summary="/root/autodl-tmp/experiments/immuneclip/week1/results/traj_${tag}_contrastive_full_s${seed}_summary.json"
  if [[ -s "${summary}" ]]; then
    echo "[skip rebound] ${tag} seed=${seed} already has ${summary}"
    return
  fi
  python scripts/run_downstream.py \
    --ckpt "${ckpt}" \
    --ft full \
    --objective contrastive \
    --steps 300 \
    --eval_steps "${EVAL_STEPS}" \
    --batch_size 16 \
    --lr 1e-6 \
    --seed "${seed}" \
    --tag "${tag}" \
    --subset 1000 \
    --downstream cc3m \
    --cc3m_root "${CC3M_ROOT}" \
    --cc3m_csv "${CLEAN_CSV}" \
    --revival_threshold 0.5 \
    2>&1 | tee "${BATCH_LOG_DIR}/downstream_${tag}_s${seed}.log"
}

run_variant() {
  local variant="$1"
  local seed="$2"
  shift 2
  local tag="formal_sp_${variant}_s${seed}"
  local out="${RUN_ROOT}/${tag}"
  local ckpt="${out}/checkpoints/${tag}_final.pt"
  mkdir -p "${out}/logs"
  if [[ -s "${ckpt}" ]]; then
    echo "[skip train] ${tag} already has ${ckpt}"
  else
    python scripts/immuneclip_new_train.py \
      "${train_common[@]}" \
      --out_dir "${out}" \
      --tag "${tag}" \
      --seed "${seed}" \
      "$@" \
      2>&1 | tee "${out}/logs/console.log"
  fi
  run_downstream_for_ckpt "${ckpt}" "${tag}_rebound_full_ft_strict" "${seed}"
}

for seed in ${SEEDS}; do
  run_variant anchor_kd "${seed}" \
    --lambda_dir 0.0 \
    --lambda_reach 0.0 \
    --reach_steps 0 \
    --reach_mode traj_global

  run_variant dir_set "${seed}" \
    --lambda_reach 0.0 \
    --reach_steps 0 \
    --reach_mode traj_global

  run_variant traj_approx "${seed}" \
    --lambda_reach 0.15 \
    --reach_steps 1 \
    --reach_mode traj_global

  run_variant checkpoint_rho "${seed}" \
    --lambda_reach 0.15 \
    --reach_steps 1 \
    --reach_mode checkpoint_rho \
    --virtual_optimizer sgd \
    --virtual_lrs 5e-7,1e-6
done

python - <<'PY'
import json, glob, os, re
rows = []
pattern = '/root/autodl-tmp/experiments/immuneclip/week1/results/traj_formal_sp_*_rebound_full_ft_strict_contrastive_full_s*_summary.json'
for path in sorted(glob.glob(pattern)):
    d = json.load(open(path))
    rows.append({
        'tag': d.get('tag'),
        'asr_step0': d.get('asr_step0'),
        'asr_final': d.get('asr_final'),
        'asr_max': d.get('asr_max'),
        'rebound_delta': d.get('rebound_delta'),
        'aurc_asr': d.get('aurc_asr'),
        'ca_step0': d.get('ca_step0'),
        'ca_final': d.get('ca_final'),
        'eval_steps': d.get('eval_steps'),
        'summary': path,
    })
out = '/root/autodl-tmp/experiments/immuneclip-new/results/single_proxy_formal_rebound_summary.json'
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(rows, open(out, 'w'), indent=2)
print(json.dumps(rows, indent=2))
PY
