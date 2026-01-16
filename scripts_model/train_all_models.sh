#!/usr/bin/env bash
set -euo pipefail

PY=python3
cd "$(dirname "$0")"

MODELS=(MLP CNN CONVMIXER RESNET LSTM GRU TCN TRANSFORMER KALMANNET)

echo "========================================="
echo "Train all models"
echo "========================================="

for m in "${MODELS[@]}"; do
  echo
  echo "[TRAIN] $m"
  $PY train_models.py --model "$m"
done

echo
echo "DONE (TRAIN)"
