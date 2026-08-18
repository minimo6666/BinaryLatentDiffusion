#!/usr/bin/env bash
set -Eeuo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/minimo/.conda/envs/BLD/bin/python}"
GPU_ID="${GPU_ID:-4}"
mkdir -p "${EXPERIMENT_DIR}/logs"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${EXPERIMENT_DIR}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" "${EXPERIMENT_DIR}/eval.py" \
    --snrs -15 -10 -5 0 5 10 15 20 \
    >> "${EXPERIMENT_DIR}/logs/eval.log" 2>&1
"${PYTHON_BIN}" "${EXPERIMENT_DIR}/plot.py"
