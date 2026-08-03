#!/usr/bin/env bash
set -euo pipefail

ART=/root/autodl-tmp/experiments/artifacts
PY=/root/miniconda3/envs/aaai/bin/python
SEED=42

BADCLIP_ROOT="$ART/baselines/attack_baselines/BadCLIP_GradAlign"
CC3M_ROOT=/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K
TRAIN_CSV="$CC3M_ROOT/train.csv"
BACKDOOR_CSV="$CC3M_ROOT/backdoor_banana_badnet_rs_fixed_random_16_500000_50000.csv"
IMAGENET_ROOT=/root/autodl-tmp/datasets/imagenet1k_badclip/validation
LOG_ROOT="$ART/logs/E1_badnet_rs_fixed_attack"
RUN_NAME=rn50_badnet_rs_fixed_p10pct_targetce_l10_lr2e6_ep3_s42

wait_for_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    if [[ -z "$used" || "$used" -lt 20000 ]]; then
      break
    fi
    echo "[$(date)] GPU busy (${used} MiB); waiting before BadNet-RS-Fixed attack" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
    sleep 120
  done
}

mkdir -p "$LOG_ROOT"
cd "$BADCLIP_ROOT"

echo "[$(date)] START BadNet-RS-Fixed data/attack preparation" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
if [[ ! -f "$BACKDOOR_CSV" ]]; then
  "$PY" -m backdoor.create_backdoor_data \
    --train_data "$TRAIN_CSV" \
    --templates "$IMAGENET_ROOT/classes.py" \
    --size_train_data 500000 \
    --num_backdoor 50000 \
    --patch_type badnet_rs_fixed \
    --patch_location random \
    --patch_size 16 \
    --label banana \
    2>&1 | tee "$LOG_ROOT/data_prepare.log"
fi

if [[ -f "$LOG_ROOT/poison_logs/$RUN_NAME/checkpoints/epoch_3.pt" ]]; then
  echo "[$(date)] SKIP BadNet-RS-Fixed checkpoint exists" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
  exit 0
fi

wait_for_gpu
echo "[$(date)] START $RUN_NAME" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
"$PY" -u src/main.py \
  --name "$RUN_NAME" \
  --logs "$LOG_ROOT/poison_logs" \
  --train_data "$BACKDOOR_CSV" \
  --batch_size 128 \
  --lr 2e-6 \
  --epochs 3 \
  --num_warmup_steps 500 \
  --complete_finetune \
  --pretrained \
  --image_key image \
  --caption_key caption \
  --eval_data_type ImageNet1K \
  --eval_test_data_dir "$IMAGENET_ROOT" \
  --add_backdoor \
  --asr \
  --label banana \
  --patch_size 16 \
  --patch_type badnet_rs_fixed \
  --patch_location random \
  --backdoor_target_loss \
  --backdoor_target_mode zeroshot \
  --lambda_backdoor_target 10.0 \
  --seed "$SEED" \
  2>&1 | tee "$LOG_ROOT/${RUN_NAME}.log"
echo "[$(date)] DONE $RUN_NAME" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
