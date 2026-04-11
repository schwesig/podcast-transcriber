#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

echo "==> Checking Python..."
$PYTHON --version

echo "==> Checking ffmpeg..."
if ! command -v ffmpeg &>/dev/null; then
  echo "ERROR: ffmpeg not found. Install with: sudo dnf install -y ffmpeg"
  exit 1
fi
ffmpeg -version 2>&1 | head -1

echo "==> Creating virtualenv at $VENV_DIR..."
$PYTHON -m venv "$VENV_DIR"

echo "==> Installing Python dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

echo ""
echo "✓ Setup complete. Activate with: source .venv/bin/activate"
