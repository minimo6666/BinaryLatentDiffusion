#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${EXPERIMENT_DIR}/config.sh"
PROJECT_DIR="${SOURCE_PROJECT_DIR}"

mkdir -p "${RESULT_DIR}/logs"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PROJECT_DIR}:${EXPERIMENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" "${EXPERIMENT_DIR}/train.py" \
  --jscc-checkpoint "${JSCC25_CHECKPOINT}" \
  --result-dir "${RESULT_DIR}" \
  --train-steps "${TRAIN_STEPS}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --save-every "${SAVE_EVERY}" \
  --qam-order "${QAM_ORDER}" \
  --snr-min "${SNR_MIN}" \
  --snr-max "${SNR_MAX}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --num-heads "${NUM_HEADS}" \
  --num-layers "${NUM_LAYERS}" \
  --preview-images "${PREVIEW_IMAGES}" \
  --preview-snrs "${PREVIEW_SNRS}" \
  2>&1 | tee "${RESULT_DIR}/logs/train_console.log"

"${PYTHON_BIN}" "${EXPERIMENT_DIR}/evaluate.py" \
  --checkpoint "${RESULT_DIR}/checkpoints/latest.pt" \
  --jscc-checkpoint "${JSCC25_CHECKPOINT}" \
  --reference-psnr "${REFERENCE_PSNR}" \
  --result-dir "${RESULT_DIR}" \
  --eval-snrs "${EVAL_SNRS}" \
  --num-images "${NUM_EVAL_IMAGES}" \
  --save-images "${NUM_SAVED_IMAGES}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --denoise-mode "${DENOISE_MODE}" \
  2>&1 | tee "${RESULT_DIR}/logs/eval_console.log"

"${PYTHON_BIN}" "${EXPERIMENT_DIR}/plot_results.py" \
  --metrics "${RESULT_DIR}/metrics/paired_metrics.csv" \
  --output-dir "${RESULT_DIR}/plots" \
  2>&1 | tee "${RESULT_DIR}/logs/plot_console.log"

echo "Experiment complete: ${RESULT_DIR}"
