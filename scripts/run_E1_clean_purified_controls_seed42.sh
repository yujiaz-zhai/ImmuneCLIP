#!/usr/bin/env bash
set -euo pipefail

ART=/root/autodl-tmp/experiments/artifacts
PY=/root/miniconda3/envs/aaai/bin/python
TORCHRUN=/root/miniconda3/envs/aaai/bin/torchrun
SEED=42

CLEAN_CKPT=/root/autodl-tmp/checkpoints/clip-clean-pretrained/RN50.pt
OUT_ROOT="$ART/outputs/E1_clean_purified_controls"
LOG_ROOT="$ART/logs/E1_clean_purified_controls"
CKPT_ROOT="$OUT_ROOT/checkpoints"
CC3M_ROOT=/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K
CC3M_STRICT="$CC3M_ROOT/cc3m_natural_10K_no_banana_strict.csv"
PAR_DATA_ROOT=/root/autodl-tmp/experiments/immuneclip_week2/defense_align_ep10/par_strict
PAR_ROOT=/root/workspace/usenix/baselines/PerturbAndRecover
INV_ROOT="$ART/baselines/defense_baselines/InverTune"

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$CKPT_ROOT"

run_downstream() {
  local tag="$1"
  local ckpt="$2"
  local summary="$OUT_ROOT/results/traj_${tag}_contrastive_full_s${SEED}_summary.json"
  if [[ -f "$summary" ]]; then
    echo "[$(date)] SKIP downstream $tag" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
    return
  fi
  cd "$ART/immuneclip"
  export PYTHONPATH="$ART/immuneclip:/root/workspace/usenix/scripts:${PYTHONPATH:-}"
  export IMMUNECLIP_EXP_ROOT="$OUT_ROOT"
  echo "[$(date)] START downstream $tag" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
  "$PY" run_downstream.py \
    --ckpt "$ckpt" \
    --tag "$tag" \
    --ft full \
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
    --patch_type ours_tnature \
    --patch_location middle \
    --patch_size 16 \
    --patch_name opti_patches/badCLIP.jpg \
    2>&1 | tee "$LOG_ROOT/downstream_${tag}.log"
}

run_par_clean() {
  local par_ckpt="$CKPT_ROOT/clean_par_rn50.pt"
  if [[ -f "$par_ckpt" ]]; then
    echo "[$(date)] SKIP clean PAR checkpoint exists: $par_ckpt" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
    return
  fi
  cd "$PAR_ROOT"
  local par_log="$LOG_ROOT/par_clean_train.log"
  echo "[$(date)] START PAR on clean RN50" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
  MASTER_PORT=29541 "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
    --dataset cc3m \
    --load-pretrained-clip "$CLEAN_CKPT" \
    --model RN50 \
    --backdoor-tuple 0,badclip,16,middle,0.5,banana \
    --train-data "$PAR_DATA_ROOT/par_cc3m_250000_abs.csv" \
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
    --addendum clean_control_s42 \
    --imagenet-root "$PAR_DATA_ROOT/imagenet_imagefolder_5pc" \
    2>&1 | tee "$par_log"
  local final
  final="$(grep 'FINAL_STRING:' "$par_log" | tail -1 | sed 's/.*FINAL_STRING://')"
  if [[ -z "$final" || ! -f "$final" ]]; then
    echo "[$(date)] ERROR clean PAR checkpoint not found" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
    exit 3
  fi
  cp -f "$final" "$par_ckpt"
}

write_invertune_config() {
  "$PY" - "$INV_ROOT/config/badclip_banana_paper.yaml" "$OUT_ROOT/clean_invertune.yaml" "$CLEAN_CKPT" "$OUT_ROOT/invertune_clean_s42" <<'PY'
import sys, yaml
src, dst, ckpt, root = sys.argv[1:5]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["model"]["model_path"] = ckpt
cfg["output"]["root"] = root
cfg["inversion"]["trigger_path"] = f"{root}/inversion/latest.pth"
cfg["evaluation"]["checkpoint"] = f"{root}/checkpoints/defended_model.pt"
cfg["evaluation"]["output"] = f"{root}/evaluation/defended_metrics.json"
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(dst)
PY
}

run_invertune_clean() {
  local ckpt="$OUT_ROOT/invertune_clean_s42/checkpoints/defended_model.pt"
  if [[ -f "$ckpt" ]]; then
    echo "[$(date)] SKIP clean InverTune checkpoint exists: $ckpt" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
    return
  fi
  write_invertune_config
  cd "$INV_ROOT"
  echo "[$(date)] START InverTune on clean RN50" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
  "$PY" -u BadCLIPEvaluate.py --config "$OUT_ROOT/clean_invertune.yaml" --checkpoint "$CLEAN_CKPT" --output "$OUT_ROOT/invertune_clean_s42/evaluation/baseline_metrics.json" \
    2>&1 | tee "$LOG_ROOT/invertune_clean_baseline.log"
  "$PY" -u BadCLIPTriggerInversionPaper.py --config "$OUT_ROOT/clean_invertune.yaml" \
    2>&1 | tee "$LOG_ROOT/invertune_clean_trigger_inversion.log"
  "$PY" -u BadCLIPActivationTuningPaper.py --config "$OUT_ROOT/clean_invertune.yaml" \
    2>&1 | tee "$LOG_ROOT/invertune_clean_activation_tuning.log"
  "$PY" -u BadCLIPEvaluate.py --config "$OUT_ROOT/clean_invertune.yaml" --checkpoint "$ckpt" --output "$OUT_ROOT/invertune_clean_s42/evaluation/defended_metrics.json" \
    2>&1 | tee "$LOG_ROOT/invertune_clean_defended.log"
}

run_par_clean
run_invertune_clean
run_downstream rn50_clean_par_full_s42 "$CKPT_ROOT/clean_par_rn50.pt"
run_downstream rn50_clean_invertune_full_s42 "$OUT_ROOT/invertune_clean_s42/checkpoints/defended_model.pt"
echo "[$(date)] DONE clean purified controls" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
