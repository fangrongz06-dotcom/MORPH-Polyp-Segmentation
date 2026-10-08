#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${DATA_ROOT:?Set DATA_ROOT to your dataset directory}"
: "${CHECKPOINT:?Set CHECKPOINT to a trained MORPH checkpoint}"
python test.py --config configs/morph_1m.yaml --data_root "$DATA_ROOT" --checkpoint "$CHECKPOINT" "$@"
