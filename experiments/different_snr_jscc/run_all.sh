#!/usr/bin/env bash
set -Eeuo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${EXPERIMENT_DIR}/train_all.sh"
"${EXPERIMENT_DIR}/eval_all.sh"
