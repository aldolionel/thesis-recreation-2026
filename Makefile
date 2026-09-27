.PHONY: data all test fast

PYTHON ?= python

data:
	$(PYTHON) scripts/run_all.py --only data

all:
	$(PYTHON) scripts/run_all.py

test:
	$(PYTHON) scripts/run_all.py --only test

fast:
	$(PYTHON) scripts/run_all.py --fast
