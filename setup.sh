#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

echo "==> Checking Python..."
$PYTHON --version
# pyproject requires >=3.11; fail with a clear message instead of a pip resolver error
if ! $PYTHON -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "ERROR: Python 3.11+ required (see requires-python in pyproject.toml)."
  echo "       Select another interpreter with: PYTHON=python3.13 ./setup.sh"
  exit 1
fi

echo "==> Checking ffmpeg..."
if ! command -v ffmpeg &>/dev/null; then
  echo "ERROR: ffmpeg not found. Install it with:"
  # Detect the package manager rather than guessing from the distro name
  if command -v brew    &>/dev/null; then echo "  brew install ffmpeg"
  elif command -v dnf   &>/dev/null; then echo "  sudo dnf install -y ffmpeg-free   (or enable RPM Fusion for full ffmpeg)"
  elif command -v apt   &>/dev/null; then echo "  sudo apt install -y ffmpeg"
  elif command -v pacman &>/dev/null; then echo "  sudo pacman -S ffmpeg"
  elif command -v zypper &>/dev/null; then echo "  sudo zypper install ffmpeg"
  elif command -v apk   &>/dev/null; then echo "  sudo apk add ffmpeg"
  else echo "  install ffmpeg with your system package manager"
  fi
  exit 1
fi
ffmpeg -version 2>&1 | head -1

# mlx-whisper is arm64-macOS only. Ask $PYTHON rather than the shell: an
# x86_64 interpreter (Rosetta, Intel Homebrew) can run on an arm64 host, and
# installing the mlx stack for it would fail.
EXTRAS=""
if [ "$($PYTHON -c 'import platform; print(platform.system(), platform.machine())')" = "Darwin arm64" ]; then
  EXTRAS="[mlx]"
  echo "==> Apple Silicon detected, including the mlx backend"
fi

echo "==> Creating virtualenv at $VENV_DIR..."
$PYTHON -m venv "$VENV_DIR"

echo "==> Installing Python dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -e ".${EXTRAS}"
"$VENV_DIR/bin/pip" install "pytest>=8.0" "pytest-mock>=3.0"

echo ""
echo "✓ Setup complete. Activate with: source .venv/bin/activate"
