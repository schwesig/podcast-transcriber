#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

echo "==> Checking Python..."
$PYTHON --version

echo "==> Checking ffmpeg..."
if ! command -v ffmpeg &>/dev/null; then
  echo "ERROR: ffmpeg not found. Install it with:"
  case "$(uname -s)" in
    Darwin) echo "  brew install ffmpeg" ;;
    *)      echo "  sudo dnf install -y ffmpeg-free   (or enable RPM Fusion for full ffmpeg)" ;;
  esac
  exit 1
fi
ffmpeg -version 2>&1 | head -1

# mlx-whisper runs on the Apple Silicon GPU and is a no-op elsewhere
EXTRAS=""
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  EXTRAS="[mlx]"
  echo "==> Apple Silicon detected — including the mlx backend"
fi

echo "==> Creating virtualenv at $VENV_DIR..."
$PYTHON -m venv "$VENV_DIR"

echo "==> Installing Python dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -e ".${EXTRAS}"
"$VENV_DIR/bin/pip" install "pytest>=8.0" "pytest-mock>=3.0"

echo ""
echo "✓ Setup complete. Activate with: source .venv/bin/activate"
