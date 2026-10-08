#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${DATA_ROOT:?Set DATA_ROOT to your dataset directory}"
for seed in ${SEEDS:-1234 2345 3456}; do
  python train.py --config configs/morph_1m.yaml --data_root "$DATA_ROOT" --seed "$seed" "$@"
done
