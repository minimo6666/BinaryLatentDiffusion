#!/usr/bin/env bash
set -Eeuo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/minimo/.conda/envs/BLD/bin/python}"
GPU_ID="${GPU_ID:-4}"
TRAIN_STEPS="${TRAIN_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
mkdir -p "${EXPERIMENT_DIR}/logs"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${EXPERIMENT_DIR}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" "${EXPERIMENT_DIR}/train.py" \
    --train-steps "${TRAIN_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --snr-min -15 --snr-max 20 --qam-order 16 \
    >> "${EXPERIMENT_DIR}/logs/train.log" 2>&1
