#!/usr/bin/env bash
set -euo pipefail

ART=/root/autodl-tmp/experiments/artifacts
BADCLIP=$ART/baselines/attack_baselines/BadCLIP_GradAlign
PY=/root/miniconda3/envs/aaai/bin/python
SEED=42

DATA_ROOT=/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K
TRAIN_CSV=$DATA_ROOT/train.csv
POISON_CSV=$DATA_ROOT/backdoor_banana_random_random_16_500000_1500.csv
IMAGENET_VAL=/root/autodl-tmp/datasets/imagenet1k_badclip/validation

RUN_NAME=rn50_badnet_random_poison_ep10_s42
LOG_ROOT=$ART/logs/E1_badnet_attack

mkdir -p "$LOG_ROOT"
cd "$BADCLIP"

echo "[$(date)] START badnet data/attack preparation" | tee -a "$LOG_ROOT/master_seed${SEED}.log"

if [[ ! -f "$POISON_CSV" ]]; then
  "$PY" -m backdoor.create_backdoor_data \
    --train_data "$TRAIN_CSV" \
    --templates "$IMAGENET_VAL/classes.py" \
    --size_train_data 500000 \
    --num_backdoor 1500 \
    --patch_type random \
    --patch_location random \
    --patch_size 16 \
    --label banana \
    2>&1 | tee "$LOG_ROOT/create_badnet_data_seed${SEED}.log"
else
  echo "[$(date)] reuse existing $POISON_CSV" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
fi

echo "[$(date)] START $RUN_NAME" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
"$PY" -u src/main.py \
  --name "$RUN_NAME" \
  --logs "$LOG_ROOT/poison_logs" \
  --train_data "$POISON_CSV" \
  --batch_size 128 \
  --lr 1e-6 \
  --epochs 10 \
  --num_warmup_steps 10000 \
  --complete_finetune \
  --pretrained \
  --image_key image \
  --caption_key caption \
  --eval_data_type ImageNet1K \
  --eval_test_data_dir "$IMAGENET_VAL" \
  --add_backdoor \
  --asr \
  --label banana \
  --patch_size 16 \
  --patch_type random \
  --patch_location random \
  --seed "$SEED" \
  2>&1 | tee "$LOG_ROOT/${RUN_NAME}.log"

echo "[$(date)] DONE $RUN_NAME" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
