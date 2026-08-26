#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="/home/user/venvs/parksoon-tf"
PYTHON_BIN="$VENV_DIR/bin/python"
NVIDIA_ROOT="$VENV_DIR/lib/python3.12/site-packages/nvidia"
NVIDIA_LIBS="$(find "$NVIDIA_ROOT" -type d -name lib -printf '%p:' 2>/dev/null)"

export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL="1"

if [[ "${1:-}" == "--check" ]]; then
    exec "$PYTHON_BIN" -c \
        "import cv2, streamlit, tensorflow as tf; print('TensorFlow', tf.__version__); print('GPU', tf.config.list_physical_devices('GPU')); print('OpenCV', cv2.__version__); print('Streamlit', streamlit.__version__)"
fi

cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" -m streamlit run kiosk.py \
    --server.headless true \
    --server.address 127.0.0.1 \
    "${@}"
