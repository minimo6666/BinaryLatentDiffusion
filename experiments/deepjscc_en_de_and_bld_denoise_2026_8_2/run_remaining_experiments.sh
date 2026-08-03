#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${EXPERIMENT_DIR}/config.sh"
mkdir -p "${RESULT_DIR}/logs"

echo "Wave 1: JSCC -15 on GPU 4; JSCC 0 on GPU 5; JSCC 4 on GPU 6"
bash "${EXPERIMENT_DIR}/run_one_base_jscc.sh" -15 4 &
PID_M15=$!
bash "${EXPERIMENT_DIR}/run_one_base_jscc.sh" 0 5 &
PID_0=$!
bash "${EXPERIMENT_DIR}/run_one_base_jscc.sh" 4 6 &
PID_4=$!
wait "${PID_M15}"
wait "${PID_0}"
wait "${PID_4}"

echo "Wave 2: JSCC 8 on GPU 4; add -15 dB test point to JSCC 25 on GPU 5"
bash "${EXPERIMENT_DIR}/run_one_base_jscc.sh" 8 4 &
PID_8=$!
(
  export CUDA_VISIBLE_DEVICES=5
  export PYTHONPATH="${SOURCE_PROJECT_DIR}:${EXPERIMENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  "${PYTHON_BIN}" "${EXPERIMENT_DIR}/evaluate.py" \
    --checkpoint "${RESULT_DIR}/checkpoints/latest.pt" \
    --jscc-checkpoint "${JSCC25_CHECKPOINT}" \
    --reference-psnr "${REFERENCE_PSNR}" \
    --result-dir "${RESULT_DIR}" \
    --eval-snrs=-15,0,4,8,12,15,25 \
    --num-images "${NUM_EVAL_IMAGES}" \
    --save-images "${NUM_SAVED_IMAGES}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --denoise-mode "${DENOISE_MODE}" \
    2>&1 | tee "${RESULT_DIR}/logs/eval_console.log"
) &
PID_25_EVAL=$!
wait "${PID_8}"
wait "${PID_25_EVAL}"

export PYTHONPATH="${SOURCE_PROJECT_DIR}:${EXPERIMENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" "${EXPERIMENT_DIR}/plot_results.py" \
  --metrics "${RESULT_DIR}/metrics/paired_metrics.csv" \
  --output-dir "${RESULT_DIR}/plots" \
  2>&1 | tee "${RESULT_DIR}/logs/plot_console.log"
"${PYTHON_BIN}" "${EXPERIMENT_DIR}/aggregate_results.py" \
  --results-dir "${RESULT_DIR}" \
  2>&1 | tee "${RESULT_DIR}/logs/aggregate_console.log"

"${PYTHON_BIN}" "${EXPERIMENT_DIR}/plot_all_jscc_bld_overlay.py" \
  --results-dir "${RESULT_DIR}" \
  2>&1 | tee "${RESULT_DIR}/logs/overlay_plot_console.log"

date --iso-8601=seconds > "${RESULT_DIR}/remaining_experiments.done"
echo "All SNR-specific JSCC + BLD experiments completed."
