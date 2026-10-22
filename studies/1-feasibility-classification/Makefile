PYTHON ?= python3
EXPERIMENT_PYTHON ?= python
.DEFAULT_GOAL := help
.PHONY: help plan prepare check test integrity
help:
	@echo "Completed feasibility study. Read the report and REPRODUCIBILITY.md. Checks: make check, make test, make integrity."
plan:
	$(PYTHON) -B 02-Code/prepare.py
prepare:
	$(PYTHON) -B 02-Code/prepare.py --write
check:
	$(PYTHON) -B 02-Code/prepare.py --check
	$(PYTHON) -B 02-Code/verify.py
test:
	PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(EXPERIMENT_PYTHON) -B 02-Code/test_experiment.py
integrity:
	$(PYTHON) -B 02-Code/verify_release.py
