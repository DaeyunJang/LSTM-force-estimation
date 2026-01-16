@echo off
setlocal enabledelayedexpansion

set PY=python
cd /d %~dp0

set TEST_CSV=../datasets/test/data_*.csv
set TEST_JSON=../datasets/test/curve_fit_result-joint_angle_*.json
set RESULTS_ROOT=../results

set MODELS=MLP CNN CONVMIXER RESNET LSTM GRU TCN TRANSFORMER KALMANNET

echo =========================================
echo Train all models
echo =========================================

for %%M in (%MODELS%) do (
    echo.
    echo [TRAIN] %%M
    %PY% train_models.py --model %%M
    if errorlevel 1 (
        echo [ERROR] Training failed for %%M
        exit /b 1
    )
)

echo.
echo =========================================
echo Predict using latest run per model
echo =========================================

for %%M in (%MODELS%) do (
    set FIT_ROOT=../fit/fit_%%M

    echo.
    echo [LATEST] %%M
    for /f "delims=" %%D in ('%PY% find_latest_run.py !FIT_ROOT!') do set LATEST_DIR=%%D

    echo [PREDICT] %%M -> !LATEST_DIR!
    %PY% predict_models.py --model_dir "!LATEST_DIR!" --test_csv "%TEST_CSV%" --test_json "%TEST_JSON%" --save_root "%RESULTS_ROOT%"
    if errorlevel 1 (
        echo [ERROR] Predict failed for %%M
        exit /b 1
    )
)

echo.
echo DONE
endlocal
