#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${HLS_CONDA_ENV:-Bench4HLS}"

conda run -n "${ENV_NAME}" python -m pip install \
  tomli \
  requests \
  matplotlib \
  seaborn \
  pytest

conda run -n "${ENV_NAME}" python - <<'PY'
import sys
import matplotlib
import pytest
import requests
import seaborn
import tomli

assert sys.version_info[:2] == (3, 10), sys.version
print("python", sys.version)
for module in (tomli, requests, matplotlib, seaborn, pytest):
    print(module.__name__, getattr(module, "__version__", "unknown"))
PY
