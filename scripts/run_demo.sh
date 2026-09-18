#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  printf '%s\n' 'Create .venv and install the project using the README commands first.' >&2
  exit 1
fi
"$PYTHON" -m asofcast verify --artifacts artifacts/demo
exec "$PYTHON" -m asofcast serve --artifacts artifacts/demo --port "${1:-8000}"
