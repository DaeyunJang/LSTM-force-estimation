#!/usr/bin/env bash
set -euo pipefail

PY=python3
cd "$(dirname "$0")"

TEST_CSV="../datasets/test/data_*.csv"
TEST_JSON="../datasets/test/curve_fit_result-joint_angle_*.json"
RESULTS_ROOT="../results"

MODELS=(MLP CNN CONVMIXER RESNET LSTM GRU TCN TRANSFORMER KALMANNET)
#MODELS=(MLP LSTM)

echo "========================================="
echo "Train all models"
echo "========================================="
for m in "${MODELS[@]}"; do
  echo
  echo "[TRAIN] $m"
  $PY train_models.py --model "$m"
done

echo
echo "========================================="
echo "Predict using latest run per model"
echo "========================================="
for m in "${MODELS[@]}"; do
  fit_root="../fit/fit_${m}"
  latest_dir="$($PY find_latest_run.py "$fit_root")"
  echo
  echo "[PREDICT] $m -> $latest_dir"
  $PY predict_models.py --model_dir "$latest_dir" --test_csv "$TEST_CSV" --test_json "$TEST_JSON" --save_root "$RESULTS_ROOT"
echo "-----------------------------------------"
done

echo
echo "DONE"
