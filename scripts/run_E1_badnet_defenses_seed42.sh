#!/usr/bin/env bash
set -euo pipefail

ART=/root/autodl-tmp/experiments/artifacts
PY=/root/miniconda3/envs/aaai/bin/python
SEED=42

ATTACK_LOG_ROOT="$ART/logs/E1_badnet_attack_strong/poison_logs/rn50_badnet_random_p5pct_ep5_s42"
ATTACK_CKPT="$ATTACK_LOG_ROOT/checkpoints/epoch_5.pt"
OUT_ROOT="$ART/outputs/E1_badnet_defenses"
LOG_ROOT="$ART/logs/E1_badnet_defenses"
CKPT_ROOT="$OUT_ROOT/checkpoints"
CC3M_ROOT=/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K
CC3M_TRAIN="$CC3M_ROOT/train.csv"
CC3M_STRICT="$CC3M_ROOT/cc3m_natural_10K_no_banana_strict.csv"
IMAGENET_ROOT=/root/autodl-tmp/datasets/imagenet1k_badclip/validation
IMAGENET_LABELS="$IMAGENET_ROOT/labels.csv"
PAR_ROOT=/root/workspace/usenix/baselines/PerturbAndRecover
BADCLIP_ROOT="$ART/baselines/attack_baselines/BadCLIP_GradAlign"
INV_ROOT="$ART/baselines/defense_baselines/InverTune"

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$CKPT_ROOT"

wait_for_attack() {
  echo "[$(date)] WAIT BadNet-random-5pct attack checkpoint: $ATTACK_CKPT" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
  while [[ ! -f "$ATTACK_CKPT" ]]; do
    if ! pgrep -f "src/main.py --name rn50_badnet_random_p5pct_ep5_s42" >/dev/null; then
      echo "[$(date)] ERROR BadNet-random-5pct attack process stopped before epoch_5 checkpoint" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
      exit 2
    fi
    sleep 60
  done
  echo "[$(date)] READY BadNet attack checkpoint" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
}

run_downstream() {
  local tag="$1"
  local ckpt="$2"
  local ft="$3"
  local summary="$OUT_ROOT/results/traj_${tag}_contrastive_${ft}_s${SEED}_summary.json"
  if [[ -f "$summary" ]]; then
    echo "[$(date)] SKIP downstream $tag ($ft)" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
    return
  fi
  cd "$ART/immuneclip"
  export PYTHONPATH="$ART/immuneclip:/root/workspace/usenix/scripts:${PYTHONPATH:-}"
  export IMMUNECLIP_EXP_ROOT="$OUT_ROOT"
  echo "[$(date)] START downstream $tag ft=$ft" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
  "$PY" run_downstream.py \
    --ckpt "$ckpt" \
    --tag "$tag" \
    --ft "$ft" \
    --objective contrastive \
    --steps 300 \
    --eval_steps 5,10,20,30,50,100,200,300 \
    --batch_size 16 \
    --lr 1e-6 \
    --seed "$SEED" \
    --subset 1000 \
    --downstream cc3m \
    --cc3m_root "$CC3M_ROOT" \
    --cc3m_csv "$CC3M_STRICT" \
    --revival_threshold 0.5 \
    --target_label banana \
    --patch_type random \
    --patch_location random \
    --patch_size 16 \
    --patch_name "" \
    2>&1 | tee "$LOG_ROOT/downstream_${tag}_${ft}.log"
}

prepare_par_data() {
  if [[ -f "$OUT_ROOT/par_strict/par_cc3m_250000_abs.csv" ]]; then
    return
  fi
  echo "[$(date)] START prepare PAR data" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
  mkdir -p "$OUT_ROOT/par_strict"
  "$PY" /root/workspace/usenix/scripts/immuneclip_week1_prepare_par.py \
    --cc3m-csv "$CC3M_TRAIN" \
    --cc3m-root "$CC3M_ROOT" \
    --imagenet-labels "$IMAGENET_LABELS" \
    --imagenet-root "$IMAGENET_ROOT" \
    --out-dir "$OUT_ROOT/par_strict" \
    --samples 250000 \
    --imagenet-per-class 5 \
    2>&1 | tee "$LOG_ROOT/par_prepare.log"
}

run_par() {
  local par_ckpt="$CKPT_ROOT/par_cleaned_rn50.pt"
  if [[ -f "$par_ckpt" ]]; then
    echo "[$(date)] SKIP PAR checkpoint exists: $par_ckpt" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
    return
  fi
  prepare_par_data
  cd "$PAR_ROOT"
  local par_log="$LOG_ROOT/par_train.log"
  echo "[$(date)] START PAR BadNet defense" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
  MASTER_PORT=29531 torchrun --standalone --nproc_per_node=1 train.py \
    --dataset cc3m \
    --load-pretrained-clip "$ATTACK_CKPT" \
    --model RN50 \
    --backdoor-tuple 0,random,16,random,0.5,banana \
    --train-data "$OUT_ROOT/par_strict/par_cc3m_250000_abs.csv" \
    --root "$CC3M_ROOT/images" \
    --batch-size 128 \
    --update-freq 4 \
    --epochs 2 \
    --samples 250000 \
    --workers 4 \
    --lr 3e-6 \
    --lr-start 3e-5 \
    --lr-end 1e-9 \
    --loss-thresh 2.15 \
    --output-dir "$OUT_ROOT/par_outputs" \
    --addendum badnet_random_s42 \
    --imagenet-root "$OUT_ROOT/par_strict/imagenet_imagefolder_5pc" \
    2>&1 | tee "$par_log"
  local final
  final="$(grep 'FINAL_STRING:' "$par_log" | tail -1 | sed 's/.*FINAL_STRING://')"
  if [[ -z "$final" || ! -f "$final" ]]; then
    echo "[$(date)] ERROR PAR checkpoint not found" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
    exit 3
  fi
  cp -f "$final" "$par_ckpt"
}

run_cleanclip() {
  local ckpt="$LOG_ROOT/cleanclip_logs/cleanclip_badnet_random_s42/checkpoints/epoch.pt"
  if [[ -f "$ckpt" ]]; then
    echo "[$(date)] SKIP CleanCLIP checkpoint exists: $ckpt" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
    return
  fi
  cd "$BADCLIP_ROOT"
  echo "[$(date)] START CleanCLIP BadNet defense" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
  "$PY" -u src/main.py \
    --name cleanclip_badnet_random_s42 \
    --logs "$LOG_ROOT/cleanclip_logs" \
    --checkpoint "$ATTACK_CKPT" \
    --train_data "$CC3M_STRICT" \
    --batch_size 64 \
    --num_warmup_steps 50 \
    --lr 4.5e-6 \
    --epochs 10 \
    --inmodal \
    --complete_finetune \
    --save_final \
    --eval_data_type ImageNet1K \
    --eval_test_data_dir "$IMAGENET_ROOT" \
    --add_backdoor \
    --asr \
    --label banana \
    --patch_type random \
    --patch_location random \
    --patch_size 16 \
    --device_id 0 \
    2>&1 | tee "$LOG_ROOT/cleanclip_train.log"
}

write_invertune_config() {
  "$PY" - "$INV_ROOT/config/badclip_banana_paper.yaml" "$OUT_ROOT/badnet_random_invertune.yaml" "$ATTACK_CKPT" "$OUT_ROOT/invertune_badnet_random_s42" <<'PY'
import sys, yaml
src, dst, ckpt, root = sys.argv[1:5]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["model"]["model_path"] = ckpt
cfg["output"]["root"] = root
cfg["inversion"]["trigger_path"] = f"{root}/inversion/latest.pth"
cfg["evaluation"]["checkpoint"] = f"{root}/checkpoints/defended_model.pt"
cfg["evaluation"]["output"] = f"{root}/evaluation/defended_metrics.json"
cfg["attack"]["patch_path"] = ""
cfg["attack"]["patch_location"] = "random"
cfg["attack"]["patch_size"] = 16
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(dst)
PY
}

run_invertune() {
  local ckpt="$OUT_ROOT/invertune_badnet_random_s42/checkpoints/defended_model.pt"
  if [[ -f "$ckpt" ]]; then
    echo "[$(date)] SKIP InverTune checkpoint exists: $ckpt" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
    return
  fi
  write_invertune_config
  cd "$INV_ROOT"
  echo "[$(date)] START InverTune BadNet inversion" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
  "$PY" -u BadCLIPTriggerInversionPaper.py --config "$OUT_ROOT/badnet_random_invertune.yaml" \
    2>&1 | tee "$LOG_ROOT/invertune_trigger_inversion.log"
  echo "[$(date)] START InverTune BadNet tuning" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
  "$PY" -u BadCLIPActivationTuningPaper.py --config "$OUT_ROOT/badnet_random_invertune.yaml" \
    2>&1 | tee "$LOG_ROOT/invertune_activation_tuning.log"
}

wait_for_attack
run_downstream rn50_badnet_nodef_full_s42 "$ATTACK_CKPT" full
run_par
run_cleanclip
run_invertune
run_downstream rn50_badnet_par_full_s42 "$CKPT_ROOT/par_cleaned_rn50.pt" full
run_downstream rn50_badnet_cleanclip_full_s42 "$LOG_ROOT/cleanclip_logs/cleanclip_badnet_random_s42/checkpoints/epoch.pt" full
run_downstream rn50_badnet_invertune_full_s42 "$OUT_ROOT/invertune_badnet_random_s42/checkpoints/defended_model.pt" full
run_downstream rn50_badnet_par_proj_s42 "$CKPT_ROOT/par_cleaned_rn50.pt" lora
echo "[$(date)] DONE BadNet E1 defenses" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
