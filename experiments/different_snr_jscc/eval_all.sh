#!/usr/bin/env bash
set -Eeuo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/minimo/.conda/envs/BLD/bin/python}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS_CSV}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
    echo "GPU_IDS must contain exactly four IDs" >&2
    exit 2
fi
mkdir -p "${EXPERIMENT_DIR}/logs" "${EXPERIMENT_DIR}/results"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${EXPERIMENT_DIR}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_group() {
    local gpu="$1"
    shift
    for snr in "$@"; do
        local name="${snr/-/minus}"
        local checkpoint="${EXPERIMENT_DIR}/runs/train_snr_${name}/checkpoints/jscc_train_snr_${name}_final.pt"
        [[ -s "${checkpoint}" ]] || { echo "Missing ${checkpoint}" >&2; return 1; }
        echo "[GPU ${gpu}] evaluate model trained at Eb/N0=${snr} dB"
        CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
            "${PYTHON_BIN}" "${EXPERIMENT_DIR}/eval.py" \
            --checkpoint "${checkpoint}" \
            --train-snr "${snr}" \
            --snrs -15 -10 -5 0 5 10 15 20 \
            >> "${EXPERIMENT_DIR}/logs/eval_train_snr_${name}.log" 2>&1
    done
}

run_group "${GPU_ARRAY[0]}" -15 5 &
pid0=$!
run_group "${GPU_ARRAY[1]}" -10 10 &
pid1=$!
run_group "${GPU_ARRAY[2]}" -5 15 &
pid2=$!
run_group "${GPU_ARRAY[3]}" 0 20 &
pid3=$!
wait "${pid0}" "${pid1}" "${pid2}" "${pid3}"

"${PYTHON_BIN}" "${EXPERIMENT_DIR}/plot.py"
echo "JSCC evaluation and plotting completed."
