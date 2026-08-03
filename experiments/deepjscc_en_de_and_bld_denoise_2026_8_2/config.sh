#!/usr/bin/env bash

# Every generated artifact stays below this experiment directory.
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${EXPERIMENT_DIR}/results"

# Existing trained JSCC-25 is input-only; it is never modified.
JSCC25_DIR="/mnt/data/0/mohao/Project/BLD_DSC/logs/different_snrs_jscc_en_de_ffhq/snr_25"
JSCC25_CHECKPOINT="${JSCC25_DIR}/saved_models/binaryae_ema_100000.th"
REFERENCE_PSNR="/mnt/data/0/mohao/Project/BLD_DSC/logs/different_snrs_jscc_en_de_ffhq/en_de_train_under_snr_25/psnr_summary.txt"

GPU_ID="${GPU_ID:-4}"
PYTHON_BIN="${PYTHON_BIN:-/home/minimo/miniconda3/envs/DavinciLUT/bin/python}"
SOURCE_PROJECT_DIR="${SOURCE_PROJECT_DIR:-$(cd "${EXPERIMENT_DIR}/../.." && pwd)}"

# Trial defaults for a 24 GB GPU. Increase only after the smoke test passes.
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
HIDDEN_SIZE="${HIDDEN_SIZE:-384}"
NUM_HEADS="${NUM_HEADS:-6}"
NUM_LAYERS="${NUM_LAYERS:-8}"

# Keep QAM order consistent between BLD training and paired evaluation.
QAM_ORDER="${QAM_ORDER:-16}"
SNR_MIN="${SNR_MIN:-0}"
SNR_MAX="${SNR_MAX:-15}"

# The first trial deliberately uses 100 images. Set NUM_EVAL_IMAGES=1000 for
# the final curve after confirming the direction of the gain.
EVAL_SNRS="${EVAL_SNRS:-0,4,8,12,15,25}"
NUM_EVAL_IMAGES="${NUM_EVAL_IMAGES:-100}"
NUM_SAVED_IMAGES="${NUM_SAVED_IMAGES:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
DENOISE_MODE="${DENOISE_MODE:-iterative}"
PREVIEW_IMAGES="${PREVIEW_IMAGES:-4}"
PREVIEW_SNRS="${PREVIEW_SNRS:-0,4,8,12}"
