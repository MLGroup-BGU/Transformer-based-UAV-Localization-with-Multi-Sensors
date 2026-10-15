@echo off
REM Full noise fine-tuning pipeline for one noise level: preprocess -> fine-tune decoder -> evaluate.
REM Outputs go to scripts\finetune_noise\runs\%NOISE%\ (cache, checkpoints, outputs).
REM Extra arguments are passed to finetune_decoder.py, e.g.:  run_pipeline_awgn_f100.bat --epochs 5 --lr 5e-6
setlocal

set NOISE=awgn_f100
set RAW_DIR=%~dp0..\..\Data\noise_data\noise\%NOISE%
set RUN_DIR=%~dp0runs\%NOISE%
set MAX_EVAL_EXPS=1000

set PY=%LOCALAPPDATA%\Python\bin\python.exe
if not exist "%PY%" set PY=python

echo ===== [%NOISE%] 1/3 Preprocess =====
"%PY%" "%~dp0preprocess_noise.py" --raw_dir "%RAW_DIR%" --run_dir "%RUN_DIR%" || goto :error

echo ===== [%NOISE%] 2/3 Fine-tune decoder =====
"%PY%" "%~dp0finetune_decoder.py" --raw_dir "%RAW_DIR%" --run_dir "%RUN_DIR%" %* || goto :error

echo ===== [%NOISE%] 3/3 Evaluate (Encoder-Decoder original / fine-tuned, RSSI trilateration if P_tx available) =====
"%PY%" "%~dp0eval_noise.py" --raw_dir "%RAW_DIR%" --run_dir "%RUN_DIR%" --max_exps %MAX_EVAL_EXPS% || goto :error

echo ===== [%NOISE%] Pipeline finished =====
exit /b 0

:error
echo ===== [%NOISE%] Pipeline FAILED (exit code %errorlevel%) =====
exit /b %errorlevel%
