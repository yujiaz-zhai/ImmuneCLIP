#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${ROOT}/config/badclip_banana_paper.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${1:-all}"

read_config() {
  "${PYTHON_BIN}" -c \
    'import sys, yaml; c=yaml.safe_load(open(sys.argv[1])); print(c[sys.argv[2]][sys.argv[3]])' \
    "${CONFIG}" "$1" "$2"
}

ATTACK_MODEL="$(read_config model model_path)"
OUTPUT="$(read_config output root)"

mkdir -p "${OUTPUT}/logs" "${OUTPUT}/evaluation"
cd "${ROOT}"

run_baseline() {
  "${PYTHON_BIN}" -u BadCLIPEvaluate.py \
    --config "${CONFIG}" \
    --checkpoint "${ATTACK_MODEL}" \
    --output "${OUTPUT}/evaluation/baseline_metrics.json" \
    2>&1 | tee "${OUTPUT}/logs/baseline_evaluation.log"
}

run_inversion() {
  "${PYTHON_BIN}" -u BadCLIPTriggerInversionPaper.py \
    --config "${CONFIG}" \
    2>&1 | tee "${OUTPUT}/logs/trigger_inversion.log"
}

run_tuning() {
  "${PYTHON_BIN}" -u BadCLIPActivationTuningPaper.py \
    --config "${CONFIG}" \
    2>&1 | tee "${OUTPUT}/logs/activation_tuning.log"
}

run_evaluation() {
  "${PYTHON_BIN}" -u BadCLIPEvaluate.py \
    --config "${CONFIG}" \
    --checkpoint "${OUTPUT}/checkpoints/defended_model.pt" \
    --output "${OUTPUT}/evaluation/defended_metrics.json" \
    2>&1 | tee "${OUTPUT}/logs/defended_evaluation.log"
}

case "${STAGE}" in
  baseline) run_baseline ;;
  inversion) run_inversion ;;
  tuning) run_tuning ;;
  evaluate) run_evaluation ;;
  all)
    run_baseline
    run_inversion
    run_tuning
    run_evaluation
    ;;
  *)
    echo "Usage: $0 {all|baseline|inversion|tuning|evaluate}" >&2
    exit 2
    ;;
esac
