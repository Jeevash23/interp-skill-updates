#!/usr/bin/env bash
# Study 3: run, analysis and independent verification, in that order (frozen with the protocol).
# Run from the study folder. Rerunning after an interruption resumes the run (completed networks are kept).
set -euo pipefail
PY=${PY:-../../.venv/bin/python}
WORKERS=${WORKERS:-6}
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs results
echo "pipeline start $(date -u +%FT%TZ) pid $$"
"$PY" -B code/run.py --config protocol/config.json --tasks data/tasks.json --output runs/confirmation \
    --workers "$WORKERS" --freeze FREEZE.json
echo "analysis start $(date -u +%FT%TZ)"
"$PY" -B code/analyze_study3.py --run runs/confirmation --output results
echo "verification start $(date -u +%FT%TZ)"
"$PY" -B code/verify.py --run runs/confirmation --summary results/summary.json --workers "$WORKERS"
echo "pipeline complete $(date -u +%FT%TZ)"
