#!/usr/bin/env bash
set -Eeuo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/minimo/.conda/envs/BLD/bin/python}"
TRAIN_STEPS="${TRAIN_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
QAM_ORDER="${QAM_ORDER:-16}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS_CSV}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
    echo "GPU_IDS must contain exactly four IDs" >&2
    exit 2
fi

mkdir -p "${EXPERIMENT_DIR}/logs" "${EXPERIMENT_DIR}/runs"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${EXPERIMENT_DIR}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_group() {
    local gpu="$1"
    shift
    for snr in "$@"; do
        local name="${snr/-/minus}"
        local run_dir="${EXPERIMENT_DIR}/runs/train_snr_${name}"
        local final="${run_dir}/checkpoints/jscc_train_snr_${name}_final.pt"
        mkdir -p "${run_dir}"
        if [[ -s "${final}" ]]; then
            echo "[GPU ${gpu}] skip completed Eb/N0=${snr}: ${final}"
            continue
        fi
        echo "[GPU ${gpu}] train Eb/N0=${snr} dB"
        CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
            "${PYTHON_BIN}" "${EXPERIMENT_DIR}/train.py" \
            --train-snr "${snr}" \
            --qam-order "${QAM_ORDER}" \
            --train-steps "${TRAIN_STEPS}" \
            --batch-size "${BATCH_SIZE}" \
            --learning-rate "${LEARNING_RATE}" \
            >> "${run_dir}/console.log" 2>&1
    done
}

run_group "${GPU_ARRAY[0]}" -15 5 > "${EXPERIMENT_DIR}/logs/gpu_${GPU_ARRAY[0]}.log" 2>&1 &
pid0=$!
run_group "${GPU_ARRAY[1]}" -10 10 > "${EXPERIMENT_DIR}/logs/gpu_${GPU_ARRAY[1]}.log" 2>&1 &
pid1=$!
run_group "${GPU_ARRAY[2]}" -5 15 > "${EXPERIMENT_DIR}/logs/gpu_${GPU_ARRAY[2]}.log" 2>&1 &
pid2=$!
run_group "${GPU_ARRAY[3]}" 0 20 > "${EXPERIMENT_DIR}/logs/gpu_${GPU_ARRAY[3]}.log" 2>&1 &
pid3=$!

wait "${pid0}" "${pid1}" "${pid2}" "${pid3}"
echo "All eight fixed-SNR JSCC models completed."
