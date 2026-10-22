# Study 3: a larger confirmation of Phase 2

**Result.** In 153 fresh task families (1,224 networks) the training history again made no detectable difference to
how much an update of one skill damages the other. The independent − joint difference in damage was
**+0.019 [−0.020, +0.058]** for linear networks and **+0.018 [−0.017, +0.053]** for leaky ReLU networks (97.5%
intervals, two for 95% simultaneous coverage). Under the prespecified rule both comparisons are **inconclusive**: each
interval contains zero, and each extends just beyond the +0.05 equivalence margin, by 0.008 and 0.003. A benefit of the
independent history larger than 0.020 (linear) or 0.017 (leaky) is excluded. The equivalence predicted with probability
0.80 and 0.85 was therefore not established: the family standard deviations were larger than in Phase 2 (0.211 and
0.191 against 0.174 and 0.167), and the estimates lay slightly above zero.

## Design

Frozen at **10:07:33 UTC on 10 October 2026**, before any network was trained (`FREEZE.json`: SHA-256 of the protocol,
predictions, configuration, sample-size and seed records, task data and all code). Details in `PROTOCOL.md`.

- **Same as Phase 2:** task, matched histories (the independent history again changes f alone 14 times and g alone
  8 times), networks (linear; leaky ReLU, slope 0.9), training, the nine update conditions and the competence rules.
  `code/study.py`, `code/amended_study.py` and `code/analyze.py` are byte-identical copies of Phase 2's frozen files.
- **Changed:** 153 task families (seeds 95001–95153) and initialization seeds 3701 and 3702, all new
  (`protocol/seed-check.json`: no overlap with the 100 seeds used earlier in AI-07); one primary endpoint per
  architecture (the damage difference D), with two-sided 97.5% Student t intervals.
- **Sample size:** the smallest number of families giving an 80% chance of an interval inside ±0.05 if the true
  difference is zero, at Phase 2's larger family standard deviation (0.1735): 153 (`protocol/sample-size.json`).
- **Decision rule, per architecture:** equivalence established if the interval lies inside (−0.05, +0.05); otherwise a
  difference established if it excludes zero; otherwise inconclusive (after a 90% competence requirement).

## Results

| | Linear | Leaky ReLU |
|---|---|---|
| Competent pairs | 306 of 306 | 306 of 306 |
| Damage, independent / joint | 0.690 / 0.671 | 0.688 / 0.670 |
| **D, primary (97.5%)** | **+0.019 [−0.020, +0.058]** | **+0.018 [−0.017, +0.053]** |
| Decision | inconclusive | inconclusive |
| Families favouring the independent history | 69 of 153 | 68 of 153 |
| Phase 2's own rule (six 99.17% intervals) | benefit ruled out | benefit ruled out |
| Changed-skill difference (95%) | +0.000038 [+0.000004, +0.000072] | +0.000066 [+0.000020, +0.000112] |
| Withheld-pair difference (95%) | +0.013 [−0.010, +0.037] | +0.014 [−0.007, +0.035] |
| D for f-updates (95%) | −0.038 [−0.083, +0.008] | −0.026 [−0.071, +0.018] |
| D for g-updates (95%) | +0.076 [+0.029, +0.122] | +0.062 [+0.022, +0.103] |
| f − g (95%) | −0.114 [−0.177, −0.051] | −0.089 [−0.148, −0.029] |
| Fixed-length steps, changed skill (95%) | −0.029 [−0.037, −0.021] | −0.029 [−0.038, −0.020] |
| Pooled with Phase 2, 217 families (97.5%) | +0.012 [−0.018, +0.043] | +0.015 [−0.013, +0.043] |

All other secondary results reproduce Phase 2. Every network was competent. The changed skill was learned to about
0.0005 and the damage was 0.67–0.69. The withheld-pair error equalled the damage divided by √2 (correlation 0.9999999
and 0.9995). The readout identity held to 2.7 × 10⁻¹⁵, and the frozen readout preserved the unchanged answers exactly.
Random 32-weight masks left the damage at 0.68–0.69, and readout-only updates raised it to 1.17–1.18. Unrestricted
updates reached a changed-skill error of 0.10 after 67 (linear) and 72 (leaky) steps, while no frozen-readout update
reached it within 200 steps. Of the 43 numeric predictions in `PREDICTIONS.md`, 41 held; the two that failed are the
primary equivalence predictions.

**What is new.** The direction split, seen post hoc in Phase 2 and prespecified here as a secondary result (95%
intervals without multiplicity correction), is replicated. After the independent history, updates of g do more damage
than after the joint history; the f-update estimates lie below zero with intervals that include zero; and the
difference between f-updates and g-updates excludes zero in both architectures. The study cannot say what causes the
split: with one fixed pair of schedules, how often each function changed alone (f 14 times, g 8) and the order of the
changes go together, and the more often changed function is always f, which is drawn first, while g is constructed
conditionally on f.

**Nulls.** The primary test established neither equivalence nor a difference in either architecture. Pooled with
Phase 2's 64 families, both intervals lie inside ±0.05; this estimate is descriptive, since Phase 2 was already known
when Study 3 was designed.

## Verification and reproducibility

`code/verify.py` imports no study code. It rebuilt every label and network in NumPy and recomputed every outcome from
the saved arrays of all 1,224 networks, making 136 comparisons with `results/summary.json`; all agree. Base
predictions, final predictions and the readout identity agree with the saved arrays to within 2.7 × 10⁻¹⁵, and paired
networks start from identical weights (`results/verification.json`). Before the freeze, a timing pilot on development
seeds checked that the number of workers does not change any output: all 1,312 arrays were identical bit for bit.

The run used 6 single-threaded worker processes from 10:07:41 to 10:23:27 UTC (946 s; 5,650 CPU-seconds, at most
304 MB per worker). The analysis and verification were finished at 10:23:57. `EVIDENCE-MANIFEST.json` hashes every file
of the study; `make -j1 verify` at the project root checks it, the freeze, and reruns the verification. To rerun
everything into a new folder, copy `code/`, `protocol/`, `data/`, `PROTOCOL.md`, `PREDICTIONS.md` and `FREEZE.json`
and run `bash code/pipeline.sh` there (about 16 minutes with 6 workers, 2.8 GB of output); the runner resumes after an
interruption.

| Path | Contents |
|---|---|
| `PROTOCOL.md`, `PREDICTIONS.md`, `protocol/`, `data/`, `FREEZE.json` | The frozen design |
| `code/` | Phase 2's engine and analysis (copies), the runner, analysis, verification, freeze and pipeline, all frozen |
| `tools/evidence_manifest.py` | Added after the run: writes and checks `EVIDENCE-MANIFEST.json` |
| `AMENDMENTS.md`, `provenance/` | What changed after the run, and the superseded first evidence manifest |
| `runs/confirmation/` | 1,224 network folders with every checkpoint and update, the run manifest and status |
| `results/summary.json` | Every number by name; `results/records/` holds the per-update records (Phase 2's analysis) |
| `results/verification.json` | The independent verification |
| `logs/pipeline.log` | The run, analysis and verification log |

The run, analysis and verification ran once, as frozen, with no failed job. After the run, one file was added to the
frozen `code/` folder and later moved to `tools/`, and the first evidence manifest was superseded; `AMENDMENTS.md`
records both.
