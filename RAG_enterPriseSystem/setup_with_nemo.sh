#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
elif [ -x "/usr/local/bin/python3.11" ]; then
  PYTHON_BIN="/usr/local/bin/python3.11"
elif [ -x "/opt/homebrew/bin/python3.11" ]; then
  PYTHON_BIN="/opt/homebrew/bin/python3.11"
else
  echo "Python 3.11 not found. Install it first: brew install python@3.11"
  exit 1
fi

echo "Using Python: $PYTHON_BIN"
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
else
  echo "Reusing existing .venv. Remove it manually if you want a fresh environment."
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-py311.txt
python - <<'PY'
import qdrant_client, logfire, nemoguardrails
print('✅ qdrant_client, logfire, and nemoguardrails imported successfully')
PY

echo "✅ Setup complete. Activate with: source .venv/bin/activate"
