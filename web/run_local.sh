#!/usr/bin/env bash
# Runs the subtitle demo on this PC's NVIDIA GPU at http://127.0.0.1:8000 (Ctrl+C stops it).
# The first run builds web/.venv-gpu with CUDA PyTorch (about 3 GB, 5-10 minutes); later runs start in seconds.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=web/.venv-gpu
if [ ! -f "$VENV/.installed" ]; then
  echo "Setting up $VENV (first run only)..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu128
  "$VENV/bin/pip" install -r web/requirements.txt
  touch "$VENV/.installed"
fi

"$VENV/bin/python" -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none found, running on the CPU (slow)')"

# Use the local copy of the released model when it exists; otherwise it downloads from Hugging Face (3 GB, once)
if [ -z "${BANGLA_ASR_MODEL:-}" ] && [ -d data/release/model ]; then
  export BANGLA_ASR_MODEL="$PWD/data/release/model"
fi

echo "Open http://127.0.0.1:8000 once it says 'Uvicorn running'."
PYTHONPATH=src exec "$VENV/bin/python" web/backend/local_server.py
