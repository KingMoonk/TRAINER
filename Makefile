# The short commands. Each target calls tasks.py (or pnpm in web/), which holds the real logic.
#   make setup      once per machine: .venv, the right torch (MPS / CUDA), requirements, web packages; prints the device
#   make dev        the live loop: API (--reload) + Next dev server on 127.0.0.1; Ctrl+C stops both; tailnet / -> page, /api/ -> API
#   make run        the built app: API serving web/out, opens the browser (python app.py); tailnet / -> API
#   make test       ruff + all unit tests; `make test T="tests.test_x tests.test_y"` for just those modules
#   make web-check  web/: Vitest, lint, typecheck, build
# Windows: install make once (`winget install ezwinports.make`), then the same commands work in PowerShell.

ifeq ($(OS),Windows_NT)
  VENV_PY := .venv/Scripts/python.exe
  SYS_PY  ?= py -3
else
  VENV_PY := .venv/bin/python
  SYS_PY  ?= python3
endif

T ?=

.PHONY: help setup dev run test test-ci web-test web-lint web-build web-check web-fixtures

help:
	@echo "make setup | dev | run | test [T=\"tests.test_x\"] | test-ci | web-test | web-lint | web-build | web-check | web-fixtures"

setup:
	$(SYS_PY) tasks.py setup

dev:
	$(VENV_PY) tasks.py dev

run:
	$(VENV_PY) tasks.py run

test:
	$(VENV_PY) tasks.py test $(T)

test-ci:
	$(VENV_PY) tasks.py test:ci

web-test:
	$(VENV_PY) tasks.py web:test

web-lint:
	$(VENV_PY) tasks.py web:lint

web-build:
	$(VENV_PY) tasks.py web:build

web-fixtures:
	$(VENV_PY) tasks.py web:fixtures

web-check: web-test web-lint web-build
