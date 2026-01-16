@echo off
setlocal enabledelayedexpansion

set PY=python
cd /d %~dp0

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
echo DONE (TRAIN)
endlocal
