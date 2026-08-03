#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <base-jscc-snr> <gpu-id>" >&2
  exit 2
fi

BASE_SNR="$1"
GPU_ID="$2"
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${EXPERIMENT_DIR}/config.sh"

JSCC_CHECKPOINT="/mnt/data/0/mohao/Project/BLD_DSC/logs/different_snrs_jscc_en_de_ffhq/snr_${BASE_SNR}/saved_models/binaryae_ema_100000.th"
REFERENCE_FILE="/mnt/data/0/mohao/Project/BLD_DSC/logs/different_snrs_jscc_en_de_ffhq/en_de_train_under_snr_${BASE_SNR}/psnr_summary.txt"
RUN_DIR="${RESULT_DIR}/jscc_train_snr_${BASE_SNR}"

if [[ ! -f "${JSCC_CHECKPOINT}" ]]; then
  echo "Missing JSCC checkpoint: ${JSCC_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/logs"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${SOURCE_PROJECT_DIR}:${EXPERIMENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" "${EXPERIMENT_DIR}/train.py" \
  --jscc-checkpoint "${JSCC_CHECKPOINT}" \
  --result-dir "${RUN_DIR}" \
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
  2>&1 | tee "${RUN_DIR}/logs/train_console.log"

"${PYTHON_BIN}" "${EXPERIMENT_DIR}/evaluate.py" \
  --checkpoint "${RUN_DIR}/checkpoints/latest.pt" \
  --jscc-checkpoint "${JSCC_CHECKPOINT}" \
  --reference-psnr "${REFERENCE_FILE}" \
  --result-dir "${RUN_DIR}" \
  --eval-snrs=-15,0,4,8,12,15,25 \
  --num-images "${NUM_EVAL_IMAGES}" \
  --save-images "${NUM_SAVED_IMAGES}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --denoise-mode "${DENOISE_MODE}" \
  2>&1 | tee "${RUN_DIR}/logs/eval_console.log"

echo "Completed frozen JSCC ${BASE_SNR} + BLD: ${RUN_DIR}"
