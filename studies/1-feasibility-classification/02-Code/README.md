# AI-07 execution and verification code

The CPU feasibility study is complete. [Reproducibility](../REPRODUCIBILITY.md) gives exact commands and the [report](../01-Papers/AI-07-FEASIBILITY-REPORT.md) explains the failed behavioral gate.

| File | Purpose |
|---|---|
| `prepare.py`, `support.py`, `verify.py` | Unchanged standard-library preparation and independent symbolic checks from 7 October |
| `experiment.py` | Deterministic one-worker CPU training, isolated update/control branches, raw observations and checkpoint provenance |
| `test_experiment.py` | Eight numerical and scientific invariant tests |
| `analyze.py` | Descriptive tables and the predeclared feasibility decision |
| `audit.py` | Independent label/count reconstruction, NumPy checkpoint inference and exact replay checks |
| `modular_reference.py` | Explicitly factorized positive reference with fixed modulo-eight composition |
| `audit_reference.py` | Independent verification of reference observations and preservation of the untargeted map |
| `plot_results.py` | Figures from saved observations and analyses |

`make check` still checks the original preparation snapshot and does not train. Its zero-run counts describe that historical snapshot, not current study status. `make test` checks the runner; `make integrity` checks retained evidence hashes. The default target only prints help. Scientific execution requires an explicit command and a new output directory.

The recorded interpreter is `python` when invoked from this `02-Code` directory, or `python` from the project root. It is reused without modification. The original [execution contract](EXECUTION-CONTRACT.md) describes implementation requirements; the active protocol is [the 8 October execution record](../06-Study-History/EXECUTION-2026-10-08.md). Causal-analysis adapters were not implemented because the behavioral gate failed.
