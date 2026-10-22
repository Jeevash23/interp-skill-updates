# Reproducing the AI-07 feasibility study

Run commands from the AI-07 project root. The [report](01-Papers/AI-07-FEASIBILITY-REPORT.md) describes a completed negative feasibility outcome. All three function-table families, including their withheld pairs, are development data. Reproduction does not create new independent confirmation samples.

## Recorded environment

The experiment reused `python` without installing or upgrading packages. The recorded versions are CPython 3.12.14, PyTorch 2.7.1+cpu, NumPy 2.2.0 and Matplotlib 3.10.3. The base runner uses Python and PyTorch; the plots also use NumPy and Matplotlib. Full environment information is in each run's `manifest.json`.

The environment's Python interpreter links into the bundled runtime cache; preserve that cache if reusing it. An isolated equivalent environment can instead be created elsewhere with those versions. No model download is required. The original machine has an Intel Core 5 210H; all runs use CPU float32, deterministic operations and one numerical thread. Exact bitwise replay is established on this environment; another library build or CPU may differ numerically.

## Preparation and integrity checks

```sh
make check
make integrity
make test
```

The preparation check intentionally retains the original zero-run counts; those describe the 7 October preparation snapshot. Current study status is in `06-Study-History/STATUS.json`. The integrity verifier checks all files in the final evidence inventory, execution/analysis/diagnostic freezes, original reference hashes, complete run inventories and audit totals. It performs no training. The test command runs eight bounded implementation checks with synthetic/untrained or very briefly updated networks.

## Analysis replay without model execution

Choose a fresh directory. Existing output directories are refused.

```sh
python3 -B 02-Code/analyze.py --run 04-Results/pilot-v1 --output /tmp/ai07-pilot-analysis-replay
python3 -B 02-Code/analyze.py --run 04-Results/long-final-v1 --output /tmp/ai07-long-final-analysis-replay
```

The JSON summaries and all six CSV tables should match the corresponding retained analysis exactly. They contain no analysis execution timestamp. Different output paths do not change the results. The per-run execution times inside the summary are taken from the retained status, not remeasured by analysis.

## Independent raw data and checkpoint audit

These commands load retained local checkpoints for NumPy forward verification; they do not train or modify the canonical runs. The output files must not already exist.

```sh
python -B 02-Code/audit.py --run 04-Results/pilot-v1 --analysis 04-Results/pilot-analysis-v1 --repeat 04-Results/qualification-v1 --output /tmp/ai07-pilot-audit-replay.json
python -B 02-Code/audit.py --run 04-Results/long-final-v1 --analysis 04-Results/long-final-analysis-v1 --output /tmp/ai07-long-final-audit-replay.json
python -B 02-Code/audit_reference.py --run 04-Results/modular-reference-v1 --output /tmp/ai07-reference-audit-replay.json
```

The audits verify symbolic targets, input splits, raw counts and predictions, no-update identity, branch provenance, optimizer steps, parameter differences, checkpoint outputs and paired initial states. The reference audit additionally checks bitwise preservation of the untargeted lookup map.

## Fresh scientific execution

Run these sequentially, one worker at a time. The runner checks available memory and disk before model allocation and before each job. It refuses less than 2 GiB available memory or 2 GiB disk. Keep output on project storage rather than a nearly full memory-backed temporary filesystem. Commands require new run directories; there is no overwrite or automatic resume option. Preserve any interrupted or failed output before choosing another new directory.

```sh
nice -n 10 python -B 02-Code/experiment.py --qualification --output 04-Results/replay-qualification
nice -n 10 python -B 02-Code/experiment.py --output 04-Results/replay-pilot
nice -n 10 python -B 02-Code/experiment.py --config 03-Data/config/long-final-v1.json --output 04-Results/replay-long-final
nice -n 10 python -B 02-Code/modular_reference.py --output 04-Results/replay-modular-reference
```

The scripts constrain thread pools and disable GPU visibility internally. The recorded pilot and longer-training runs took approximately 41 and 57 seconds, with peak resident memory about 298 MiB. Startup, validation and reporting time are additional. New execution should reproduce scientific arrays/metrics on the recorded environment, while timestamps, durations, run IDs, serialization bytes and potentially cross-platform floating-point details may differ.

## Figures and evidence preservation

`02-Code/plot_results.py` generates the two figures from retained analysis tables and the modular reference summary. It writes into `04-Results/figures/`; the published files are covered by the final manifest. To change a figure, preserve the original and record a new evidence version rather than silently replacing the frozen release. The same rule applies to protocol, code, input, audit and result files.

The initial source snapshot is separate from the final evidence inventory. The diagnostic was chosen after viewing pilot results and is explicitly labeled as such. No external preregistration, paper submission or future experiment was scheduled.
