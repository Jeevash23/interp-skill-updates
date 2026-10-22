# Can a training history make two skills independently updatable? Code release.
# `make help` lists the targets. PYTHON defaults to the environment that `make setup` creates.
PYTHON ?= .venv/bin/python
PY := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
export PYTHONDONTWRITEBYTECODE := 1
export OPENBLAS_NUM_THREADS ?= 1
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
.NOTPARALLEL:

S1 := studies/1-feasibility-classification
S2 := studies/2-controlled-histories

.PHONY: help setup python-check unpack check-freezes quick-check test numbers figures

help:
	@echo "make setup          create .venv with the recorded environment (downloads packages, see ENVIRONMENT.md)"
	@echo "make quick-check    CPU, under a minute, NumPy and SciPy only, no network: freeze records, every number"
	@echo "                    and table of the manuscript, and eight saved-output analyses rerun and compared"
	@echo "make test           the unit tests of phases 1 and 2 and phase 1's input checks (needs PyTorch: make setup)"
	@echo "make numbers        rewrite paper/generated/ from the saved outputs"
	@echo "make figures        draw Figures 1-4 into paper/figures/"

setup:
	uv venv --python 3.12.14 .venv
	grep -v '^torch==' requirements.txt > .venv/requirements-pypi.txt
	uv pip install --python .venv/bin/python -r .venv/requirements-pypi.txt
	uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1+cpu'

python-check:
	@$(PY) -c "import numpy, scipy" 2>/dev/null || { echo "No usable Python at $(PYTHON): run make setup, or set PYTHON=..."; exit 1; }

unpack: python-check
	$(PY) tools/unpack.py

check-freezes: unpack
	$(PY) tools/check_freezes.py

quick-check: unpack
	$(PY) tools/check_freezes.py
	$(PY) -B paper/make_numbers.py --check && echo "paper/make_numbers.py --check: every macro and table matches"
	$(PY) tools/replay_analyses.py

test: unpack
	$(PY) -B $(S1)/02-Code/test_experiment.py
	$(PY) -B $(S1)/02-Code/prepare.py --check
	$(PY) -B $(S1)/02-Code/verify.py
	$(PY) -B $(S2)/code/test_study.py
	$(PY) -B $(S2)/code/test_analysis.py

numbers: unpack
	$(PY) -B paper/make_numbers.py

figures: unpack
	$(PY) -B paper/make_figures.py
