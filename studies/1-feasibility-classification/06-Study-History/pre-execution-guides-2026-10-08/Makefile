PYTHON ?= python3
.DEFAULT_GOAL := help
.PHONY: help plan prepare check
help:
	@echo "Preparation only: make plan, make prepare, make check. No model execution target."
plan:
	$(PYTHON) -B 02-Code/prepare.py
prepare:
	$(PYTHON) -B 02-Code/prepare.py --write
check:
	$(PYTHON) -B 02-Code/prepare.py --check
	$(PYTHON) -B 02-Code/verify.py
