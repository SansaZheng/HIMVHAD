#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$PWD"
python -m pip install --no-cache-dir -q numpy==2.3.5 scipy==1.15.3 pandas==2.2.3 scikit-learn==1.7.2 threadpoolctl pytest
case "$1" in
  env)
    python -m pytest -q > "$CQ_OUTPUT_DIR/tests.txt" 2>&1
    python - <<'PY' > "$CQ_OUTPUT_DIR/env.json"
import json,torch,numpy,scipy,sklearn,sys
print(json.dumps(dict(status='passed',torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,python=sys.version),indent=2))
PY
    ;;
  smoke|full)
    python scripts/run_knob_experiment.py "$1" --output "$CQ_OUTPUT_DIR/experiment" --device cuda > "$CQ_OUTPUT_DIR/progress.log" 2>&1
    ;;
  *) exit 2 ;;
esac
