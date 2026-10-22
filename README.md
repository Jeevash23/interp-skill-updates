# Independent changes during training did not detectably reduce interference between two learned skills

Code and results for the preprint *Independent changes during training did not detectably reduce interference between two learned skills*, by Richard Jeevash (link to be added when it is posted).

This repository holds the code, frozen inputs, study records and saved outputs behind it; the saved outputs and the
frozen protocols are attached to its v1.0 release (see [Release files](#release-files)). Every number, generated
table and plotted value in the manuscript can be recomputed from the saved outputs on a CPU in under a minute,
without network access.

## Release files

The saved outputs and the frozen protocols are not in this repository's files; they are attached to its v1.0 release:

| File | Size | Contents |
|---|---|---|
| `per-run-outputs.tar.gz` | 45.3 MB | The saved outputs: the 3 compressed archives in `data/saved-outputs/`, which `make unpack` places in the study folders and checks file by file |
| `frozen-protocols.tar.gz` | 22 kB | The studies' protocols, predictions, amendments, approvals and freeze records (20 files), at their places in the folder, where the checks read them |
| `SHA256SUMS` | | Checksums of the two archives |

Download the three files into an empty folder, check them there, and unpack them into this one:

```sh
sha256sum -c SHA256SUMS
tar -xzf per-run-outputs.tar.gz -C /path/to/interp-skill-updates
tar -xzf frozen-protocols.tar.gz -C /path/to/interp-skill-updates
```

After unpacking, `sha256sum -c SHA256SUMS` in this folder checks every file that its `SHA256SUMS` lists, the
unpacked attachments included, and the checks below run as described.

## What the paper asks

If two skills often change separately while a network is trained, does the network become able to update one of
them while leaving the other intact? The studies train small dense networks to compute f(a) + g(b) for two lookup
functions on eight symbols. Two training histories see the same examples, environments and amount of target change:
in the *independent* history one function changes at a time, in the *joint* history both change together. After
training, one function is replaced and the network is updated on the eight new values alone, and the damage to the
other function is measured.

## Studies

| Manuscript | Folder | What it did |
|---|---|---|
| Classification version | `studies/1-feasibility-classification` | The first version of the task (27 ReLU networks, three histories, a longer-training diagnostic and a modular reference); it failed before any history comparison because its networks did not compose |
| Controlled histories and the first confirmation | `studies/2-controlled-histories` | Development, one amendment (the leaky ReLU), the first confirmation on 64 task families (512 networks), the 1,000-step frozen-readout updates, audits and an exact replay |
| Larger confirmation | `studies/3-larger-confirmation` | 153 new task families (1,224 networks), designed from the first confirmation's variance and frozen with numeric predictions before any run |

Each study keeps its protocol and freeze records, code, inputs and results. The classification version keeps the
internal folder names of its original layout (`02-Code`, `03-Data`, `04-Results`, `05-Notes`, `06-Study-History`).
The freezes are local hash records written by the study code, not public preregistrations. Changes after a freeze
are listed in each study's `AMENDMENTS.md`.

## What is in this folder

| Path | Contents |
|---|---|
| `studies/` | The three studies (above); after `make unpack`, also their saved outputs |
| `analysis/` | The post hoc and descriptive analyses of 9 October 2026 (`posthoc_*.py`, `duration_curves.py`) with their results, and `replay_check.py`, the original project's full replay (it needs the network arrays, see below) |
| `paper/make_numbers.py`, `paper/make_figures.py` | Every macro and generated table of the manuscript, and its four figures, from the saved outputs; `paper/generated/` holds the values the manuscript was built with |
| `provenance/relocation-inventory-2026-10-09.tsv.xz` | Every project file before and after the reorganisation of 9 October 2026 (renames only, hashed before and after); the manuscript counts its files |
| `RESULTS.md` | The findings with their numbers |
| `data/saved-outputs/` | Saved outputs as compressed archives (see its README) (attached to the release: `per-run-outputs.tar.gz`) |
| `tools/` | Unpacking, freeze checking and the replay of the saved-output analyses |
| `requirements.txt`, `ENVIRONMENT.md` | The exact environment |
| `RELEASE-CHANGES.md`, `release-changes.json` | Every file that differs from the original project, with the original's SHA-256, and every frozen file not shipped |

## Install

```sh
make setup        # .venv with CPython 3.12.14 and the pinned packages (see ENVIRONMENT.md)
```

This downloads the packages in `requirements.txt` (installed size about 1.3 GB, most of it PyTorch's CPU build and
jaxlib). The quick check needs only NumPy and SciPy; `ENVIRONMENT.md` gives a small route. Pass
`PYTHON=/path/to/python` to any `make` target to use another environment.

## Quick check (CPU, under a minute, no network)

First unpack the two release attachments into this folder (see [Release files](#release-files)): the checks read
the saved outputs and the freeze records they contain.

```sh
make quick-check
```

It unpacks the saved outputs and compressed inputs (checking every SHA-256), checks the protocol freezes of all
three studies against the files they name, recomputes every macro and generated table of the manuscript
(`paper/make_numbers.py --check`), and reruns phase 1's frozen analysis on its pilot and longer-training runs and the
six post hoc and descriptive analyses in a temporary copy, comparing each output with the shipped file byte for
byte. Expected end of the output:

```text
total: 4 changed for the release (documented), 52 unchanged
paper/make_numbers.py --check: every macro and table matches
...
8 analyses rerun from the saved records: every output identical to the shipped files
```

```sh
make test         # the unit tests of both phases and phase 1's input checks (PyTorch; make setup)
make numbers      # rewrite paper/generated/
make figures      # Figures 1-4 into paper/figures/
```

## What needs the network arrays

The studies saved every network's parameters and update trajectories (4.2 GB of arrays). The release keeps every
JSON run record, the first confirmation's initial and base parameters (which the geometry and frozen-rate analyses
read) and the 1,000-step frozen-readout runs, but not the other arrays. Three steps need them and cannot run here:
the frozen analyses of the controlled histories and of the larger confirmation (`code/analyze.py`,
`code/analyze_study3.py`), which extract their records from the arrays; the larger confirmation's independent
verification (`code/verify.py`); and the original project's replay (`analysis/replay_check.py`), which reruns the
phase 2 analyses, retrains eight networks and reruns the audits. Their saved results are in each study's `analysis/`
or `results/` folder (`results/verification.json`, `RELEASE-VERIFICATION.json`, `analysis/results/replay-2026-10-09.json`).
The networks can be trained again on a CPU:

- Larger confirmation: from `studies/3-larger-confirmation/`, `bash code/pipeline.sh` with `PY` set to the environment's
  Python runs the 1,224 networks (six worker processes, about 16 minutes, 2.8 GB), the analysis and the verification.
  Copy `code/`, `protocol/`, `data/`, `PROTOCOL.md`, `PREDICTIONS.md` and `FREEZE.json` into a new folder first: the
  study code refuses to overwrite an existing output folder.
- Controlled histories: `studies/2-controlled-histories/REPRODUCIBILITY-2026-10-09.md` gives the commands for the
  analyses, the exact model replay, the audits, the frozen-readout updates and a full retraining (512 confirmation
  networks: about 13 minutes on one CPU thread, about 1.2 GB of output).
- Classification version: `studies/1-feasibility-classification/REPRODUCIBILITY.md` (the pilot and the
  longer-training diagnostic took about 41 and 57 seconds).

Training is deterministic in float64 (float32 in the classification version) with one thread; in the recorded
environment the saved networks are reproduced bit for bit, and another CPU or library build may differ in the last
bits.

## Tables and figures

| Manuscript | Source |
|---|---|
| All numbers in the text | macros in `paper/generated/numbers.tex`, written by `paper/make_numbers.py` |
| Table 1 (matching of the histories) | `paper/generated/table-matching.tex` |
| Table 3 (competence) | `paper/generated/table-competence.tex` |
| Table 4 (primary result) | `paper/generated/table-primary.tex` |
| Table 5 (larger confirmation) | `paper/generated/table-study3.tex` |
| Table 6 (classification version) | `paper/generated/table-phase1.tex` |
| Table 7 (development) | `paper/generated/table-development.tex` |
| Table 9 (update conditions) | `paper/generated/table-conditions.tex` |
| Table 10 (sensitivity) | `paper/generated/table-sensitivity.tex` |
| Table 11 (sample size) | `paper/generated/table-samplesize.tex` |
| Tables 2 (design) and 8 (timeline) | written in the manuscript's source; the timeline's times are macros from the freeze records |
| Figures 1-4 | `paper/make_figures.py` → `paper/figures/fig-design.pdf`, `fig-families.pdf`, `fig-interventions.pdf`, `fig-trajectories.pdf` |

## What is not included

- The network arrays described above.
- The studies' written reports and their figures, notes on the literature, logs, a copy of the research area's
  overview page among phase 1's pre-execution guides, and the original project's other provenance records and
  manuscript source. Frozen files not shipped are listed with their SHA-256 in `RELEASE-CHANGES.md`. The evidence
  manifests (`FINAL-EVIDENCE-MANIFEST.json`, `EVIDENCE-MANIFEST.json`) are included as records; they list every file
  of each study, including the arrays left out.

## Changes from the original project

Files are copied byte for byte, except those listed in `RELEASE-CHANGES.md` with the original's SHA-256: records had
personal paths, other project folders' names and the names of software tools in working records replaced by placeholders (the assistants used are named under "Use of AI assistants" below), and local timestamps given in
UTC.

## Licence

MIT (`LICENSE`), covering the code and the studies' records and saved outputs.

## Study codes

The records and code name the studies by the codes they had while the work was done. Each code that occurs here,
and where that study is released:

| Code | Released in |
|---|---|
| AI-05 | `interp-joint-vs-single` |
| AI-07 | this repository |
| AI-08 | `interp-position-cue` |

## Use of AI assistants

As the preprint states:

> Language-model assistants (Claude, Anthropic; Codex, OpenAI) were used throughout this work, under the author's
> direction, to help design the studies, to write and run the code for the experiments and analyses, to draft and
> revise the text and to check the references against their online records. All reported results come from the
> saved study outputs, and the numbers in the text were generated from those outputs or checked against them.

In the working records shipped here, the names of software tools were replaced by placeholders such as
`[redacted]`; the assistants used are the ones named above.

## Citation

Please cite the preprint (the link will be added here when it is posted) and, for the code and results, this
repository; [CITATION.cff](CITATION.cff) has the details.

```bibtex
@software{interp_skill_updates_2026,
  author  = {Jeevash, Richard},
  title   = {{Independent changes during training did not detectably reduce interference between two learned skills: code and results}},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/Jeevash23/interp-skill-updates}
}
```
