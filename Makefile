# Core ML test suite for V1 (see coreml/tests). The build under test is chosen with
# V1_VARIANT / V1_PRECISION / V1_COMPUTE_UNITS (defaults: faithful / fp16 / CPU_AND_NE).
#   make test-fast      T1-T8 on FAST, < 10 min
#   make test-full      T9 (dev-clean + dev-other WER vs V1) and T10 (speed; needs an idle Mac)
#   make test-release   T11 (test-clean + test-other once, next to V1 PyTorch and Phonon-2)
#   make bench          the T10 benchmark alone
#   make summary        rebuild ~/gemma_stt/coreml/summary.md from results.jsonl
PY ?= $(HOME)/.venv-coreml/bin/python
RUN_ID := $(shell date +%Y%m%d-%H%M%S)

.PHONY: test-fast test-full test-release bench summary

test-fast:
	cd coreml && V1_RUN_ID=$(RUN_ID) $(PY) -m pytest -m fast; s=$$?; $(PY) summary.py; exit $$s

test-full:
	cd coreml && V1_RUN_ID=$(RUN_ID) $(PY) -m pytest -m full; s=$$?; $(PY) summary.py; exit $$s

test-release:
	cd coreml && V1_RUN_ID=$(RUN_ID) $(PY) -m pytest -m release; s=$$?; $(PY) summary.py; exit $$s

bench:
	cd coreml && $(PY) bench.py

summary:
	cd coreml && $(PY) summary.py
