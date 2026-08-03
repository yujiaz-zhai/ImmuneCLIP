#!/usr/bin/env bash
set -euo pipefail

ART=/root/autodl-tmp/experiments/artifacts
PY=/root/miniconda3/envs/aaai/bin/python
SEED=42

INV_ROOT="$ART/baselines/defense_baselines/InverTune"
OUT_ROOT="$ART/outputs/E1_badclip_invertune"
LOG_ROOT="$ART/logs/E1_badclip_invertune"
RUN_ROOT="$OUT_ROOT/invertune_badclip_banana_s42"
CONFIG="$OUT_ROOT/badclip_banana_artifact.yaml"
POISON_CKPT=/root/autodl-tmp/experiments/badclip_sbl/baseline_1_1_readme_strict/logs/nodefence_badCLIP/checkpoints/epoch_10.pt

CC3M_ROOT=/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K
CC3M_STRICT="$CC3M_ROOT/cc3m_natural_10K_no_banana_strict.csv"

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$RUN_ROOT"

write_config() {
  "$PY" - "$INV_ROOT/config/badclip_banana_paper.yaml" "$CONFIG" "$POISON_CKPT" "$RUN_ROOT" <<'PY'
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

run_invertune() {
  if [[ -f "$RUN_ROOT/checkpoints/defended_model.pt" ]]; then
    echo "[$(date)] SKIP InverTune checkpoint exists: $RUN_ROOT/checkpoints/defended_model.pt" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
    return
  fi
  write_config
  cd "$INV_ROOT"
  echo "[$(date)] START BadCLIP InverTune baseline" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
  "$PY" -u BadCLIPEvaluate.py --config "$CONFIG" --checkpoint "$POISON_CKPT" --output "$RUN_ROOT/evaluation/baseline_metrics.json" \
    2>&1 | tee "$LOG_ROOT/baseline_evaluation.log"
  echo "[$(date)] START BadCLIP InverTune inversion" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
  "$PY" -u BadCLIPTriggerInversionPaper.py --config "$CONFIG" \
    2>&1 | tee "$LOG_ROOT/trigger_inversion.log"
  echo "[$(date)] START BadCLIP InverTune tuning" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
  "$PY" -u BadCLIPActivationTuningPaper.py --config "$CONFIG" \
    2>&1 | tee "$LOG_ROOT/activation_tuning.log"
  echo "[$(date)] START BadCLIP InverTune final eval" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
  "$PY" -u BadCLIPEvaluate.py --config "$CONFIG" --checkpoint "$RUN_ROOT/checkpoints/defended_model.pt" --output "$RUN_ROOT/evaluation/defended_metrics.json" \
    2>&1 | tee "$LOG_ROOT/defended_evaluation.log"
}

run_rebound() {
  local summary="$OUT_ROOT/results/traj_rn50_badclip_invertune_full_s42_contrastive_full_s42_summary.json"
  if [[ -f "$summary" ]]; then
    echo "[$(date)] SKIP downstream summary exists: $summary" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
    return
  fi
  cd "$ART/immuneclip"
  export PYTHONPATH="$ART/immuneclip:/root/workspace/usenix/scripts:${PYTHONPATH:-}"
  export IMMUNECLIP_EXP_ROOT="$OUT_ROOT"
  echo "[$(date)] START BadCLIP InverTune downstream full" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
  "$PY" run_downstream.py \
    --ckpt "$RUN_ROOT/checkpoints/defended_model.pt" \
    --tag rn50_badclip_invertune_full_s42 \
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
    2>&1 | tee "$LOG_ROOT/downstream_full.log"
}

run_invertune
run_rebound
echo "[$(date)] DONE BadCLIP InverTune E1" | tee -a "$LOG_ROOT/master_seed${SEED}.log"
