#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=4 # 限制 Torch的线程，可根据 CPU 性能调整
export MKL_NUM_THREADS=4

echo "+===============================+"
echo "|   Starting Sticker Service    |"
echo "|           Reload On           |"
echo "+===============================+"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] NVIDIA GPU detected"
else
  echo "[INFO] No GPU detected, using CPU"
fi

echo "[INFO] Starting server..."

uvicorn sticker_service.app:app --reload

