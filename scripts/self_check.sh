#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT"

PY=(conda run -n ros2py310 python)
if [[ "${1:-}" == "--no-conda-run" ]]; then
  PY=(python)
  shift
fi

if [[ "${PY[0]}" == "conda" ]] && ! command -v conda >/dev/null 2>&1; then
  echo "[self-check] conda not found; install conda or run with --no-conda-run in an activated env" >&2
  exit 2
fi

echo "[self-check] repo_doctor"
"${PY[@]}" scripts/repo_doctor.py

echo "[self-check] train.py --self-check"
"${PY[@]}" train.py --self-check

echo "[self-check] infer.py --self-check"
"${PY[@]}" infer.py --self-check

echo "[self-check] game.py --self-check"
"${PY[@]}" game.py --self-check

echo "[self-check] ok"

