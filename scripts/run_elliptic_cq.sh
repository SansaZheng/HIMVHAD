#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -m pip install --no-cache-dir -q numpy==2.3.5 scipy==1.15.3 pandas==2.2.3 scikit-learn==1.7.2 threadpoolctl
scratch=$(mktemp -d "$CQ_OUTPUT_DIR/elliptic-inputs.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
curl --fail --silent --show-error --location --retry 2 --max-time 300 \
  'https://www.kaggle.com/api/v1/datasets/download/ellipticco/elliptic-data-set?datasetVersionNumber=1' \
  -o "$scratch/elliptic.zip"
python scripts/run_elliptic_holdout.py --device cpu --archive "$scratch/elliptic.zip" \
  --output "$scratch/experiment" | tee "$CQ_OUTPUT_DIR/progress.log"
mv "$scratch/experiment" "$CQ_OUTPUT_DIR/experiment"
