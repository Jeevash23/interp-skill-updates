"""Shared paths and helpers for the AI-07 analyses of saved outputs (no training, no writes outside analysis/results)."""
from __future__ import annotations

import os

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

PROJECT = Path(__file__).resolve().parents[1]
P1 = PROJECT / "studies/1-feasibility-classification"
P2 = PROJECT / "studies/2-controlled-histories"
CONF = P2 / "analysis/confirmation"
RESULTS = PROJECT / "analysis/results"
ARCHS = ("linear", "leaky")
HISTORIES = ("independent", "joint")


def read(path):
    return json.loads(Path(path).read_text())


def write(name: str, value) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def interval(values, confidence=0.95):
    """Two-sided Student t interval for the mean, as in the frozen analysis (studies/2-controlled-histories/code/analyze.py)."""
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    half = float(student_t.ppf(1 - (1 - confidence) / 2, n - 1) * sd / np.sqrt(n))
    return dict(n=n, mean=mean, sd=sd, low=mean - half, high=mean + half, confidence=confidence)


def family_differences(records, arch, arm, metric, select=lambda r: True):
    """Independent minus joint, per task family, averaging initialization seeds and update directions."""
    groups: dict = {}
    for r in records:
        if r["architecture"] == arch and r["arm"] == arm and select(r):
            groups.setdefault(r["task_seed"], {}).setdefault(r["history"], []).append(r[metric])
    return {k: float(np.mean(v["independent"]) - np.mean(v["joint"])) for k, v in sorted(groups.items())}
