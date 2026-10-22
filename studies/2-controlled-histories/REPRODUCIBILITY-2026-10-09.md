# Reproducing phase 2 (controlled histories)

This guide replaces the one written on 8 October 2026, which is kept as
`../../provenance/superseded/controlled-history-v2-REPRODUCIBILITY.md` (`REPRODUCIBILITY.md` in this folder is a link
to it): its `cd` path named a folder that does not exist, its interpreter had since been removed, and its rerun
commands wrote inside `runs/`. Every command below
writes only to a temporary folder. The files of this study are frozen and hashed by `FINAL-EVIDENCE-MANIFEST.json`;
`AMENDMENTS.md` lists what has changed around them since.

## Setup

Run everything from the AI-07 project folder (two levels above this one), on a CPU.

```sh
PY=python
S2=studies/2-controlled-histories
OUT=$(mktemp -d)
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

`$PY` is AI-05's environment, in which the study ran: CPython 3.12.14, PyTorch 2.7.1+cpu, NumPy 2.2.0, SciPy 1.16.0,
Matplotlib 3.10.3. It was removed on 8 October 2026 and rebuilt on 9 October 2026 with the same package versions
(`../../provenance/environment-2026-10-09.md`). Each run's `manifest.json` records the interpreter and library
versions it used. No GPU, model download or dataset is needed. All runs use float64, deterministic operations and one
thread; bitwise equality holds on this environment and may fail on another CPU or library build.

## Checks that read the saved outputs

```sh
make verify     # every manifest and freeze at its present path, and the original verifiers on the original layout
make test       # unit tests of both phases (this study: code/test_study.py and code/test_analysis.py)
make replay     # the analyses and audits rerun and eight models retrained, in a temporary folder (below)
```

`code/verify_release.py` locates files relative to the layout of 8 October 2026 and no longer runs in place;
`make verify` runs it unchanged on a temporary reconstruction of that layout.

## Recompute the analyses

```sh
"$PY" -B $S2/code/analyze.py --run $S2/runs/development  --output $OUT/development
"$PY" -B $S2/code/analyze.py --run $S2/runs/amendment    --output $OUT/amendment
"$PY" -B $S2/code/analyze.py --run $S2/runs/confirmation --output $OUT/confirmation
```

`summary.json`, `records.json`, `base-records.json` and `curves.json` reproduce the saved files in `analysis/`
value for value; only the `run` path inside `summary.json` and the time in `provenance.json` differ. `make replay`
does this comparison automatically and records it in `../../analysis/results/replay-<date>.json`.

## Exact model replay

```sh
nice -n 10 "$PY" -B $S2/code/replay_models.py --run $S2/runs/confirmation --output $OUT/replay-models
```

Retrains the eight confirmation networks of the first and last task family with the first initialization seed (both
architectures and histories), reruns all 18 updates of each and requires every saved array to be identical bit for
bit (1,312 arrays; under a minute). These repeats verify the code; they are not further samples.

## Independent audits

```sh
"$PY" -B $S2/code/audit.py --run $S2/runs/confirmation --analysis $S2/analysis/confirmation --output $OUT/confirmation-audit.json
"$PY" -B $S2/code/audit_duration.py --output $OUT/duration-audit.json
```

The audit re-implements the networks in NumPy without importing the study code. It reconstructs the labels, the
history matching, the predictions at every saved checkpoint and update step, the update conditions and the primary
intervals, and replays all 18 updates of a fixed first/last-family sample. The same command audits development and
amendment with their own run and analysis folders. Rerun on 9 October 2026 (`make replay`), the confirmation audit (about three minutes) and the
duration audit reproduced `analysis/confirmation-audit.json` and `analysis/duration-audit.json` apart from their
time stamps.

## Rerun the 1,000-step frozen-readout updates

```sh
nice -n 10 "$PY" -B $S2/code/reproduce_duration.py --output $OUT/duration
```

Reuses the 512 saved confirmation networks and repeats all 1,024 updates with the readout frozen for 1,000 steps
(about half a minute). This run was declared and frozen after the confirmation results were known
(`protocol/DURATION-CONTROL.md`, `protocol/DURATION-FREEZE.json`).

## Retrain every planned model

```sh
nice -n 10 "$PY" -B $S2/code/study.py run --config $S2/protocol/development.json --tasks $S2/data/development.json --output $OUT/development-run
nice -n 10 "$PY" -B $S2/code/amended_study.py run --config $S2/protocol/amendment.json --tasks $S2/data/amendment.json --output $OUT/amendment-run
nice -n 10 "$PY" -B $S2/code/amended_study.py run --config $S2/protocol/confirmation.json --tasks $S2/data/confirmation.json --output $OUT/confirmation-run
```

Run them one at a time. The runner refuses to start with less than 2 GiB of available memory or 3 GiB of free disk
where the output goes, and refuses an existing output folder. The confirmation run trained 512 networks in about
13 minutes on one CPU thread and writes about 1.2 GB; point `OUT` at a disk with room (for example
`OUT=$(mktemp -d -p ~/tmp)`). `prepare` mode instead of `run` regenerates the task data without training. Reusing the
confirmation seeds repeats this experiment; it does not supply a new independent confirmation.

## What must not be run in place

- `code/plot_results.py` and `code/plot_results_layout_fix.py` overwrite the hashed figures in `report/figures/` and
  `analysis/confirmation/descriptive-sensitivity.json`. The manuscript's figures come from `../../paper/make_figures.py`.
- `code/duration_control.py` writes `runs/duration-control/`, which already exists (it refuses); use
  `code/reproduce_duration.py` as above.
- `code/verify_release.py --freeze` would write a new manifest (it refuses while the old one exists).

## Evidence format

Each model folder in `runs/` keeps the initial weights, all 29 environment checkpoints, the base optimizer state, all
base predictions, both update directions, all nine update conditions and all 201 states of every update.
`updates.npz` has axes `[branch, arm, step, example]` in the order given by `updates.json`, and also records the
distance moved, the path length, the length of each step, the updated targets and the drift of the unchanged
function predicted by the readout identity. `endpoints.npz` stores the final weights of every condition, and
`base.json` the transitions, training lengths, losses and competence errors. `runs/*/snapshot/` holds the source code
of each run, hashed in its `manifest.json`.
