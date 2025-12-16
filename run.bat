@echo off
REM run.bat - Complete Medical-Grade Pipeline (Windows)
REM ================================================
REM Now mirrors run.sh (12 etapas) con evaluación base + ensemble/triage,
REM retraining focal opcional, TTA opcional y logging W&B opcional.

setlocal enabledelayedexpansion

REM Toggles (set ENABLE_WANDB_PIPELINE=1 or ENABLE_TTA=1 before running to enable)
if not defined ENABLE_WANDB_PIPELINE set ENABLE_WANDB_PIPELINE=0
if not defined ENABLE_TTA set ENABLE_TTA=0
set LOG_WANDB_ARG=
if "%ENABLE_WANDB_PIPELINE%"=="1" set LOG_WANDB_ARG=--log-wandb
set EVAL_TTA_ARGS=
if "%ENABLE_TTA%"=="1" set EVAL_TTA_ARGS=--use-tta

REM Get script directory
cd /d "%~dp0"

REM Add project root to PYTHONPATH so Python can find the 'src' module
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Disable oneDNN logs
set TF_ENABLE_ONEDNN_OPTS=0

REM Virtual environment settings
set VENV_DIR=.venv
set REQUIREMENTS_FILE=requirements.txt
set CONFIG_FILE=configs\config.yaml
set EXTERNAL_DATA_MEDICAL=data\external_navoneel_medical
set REPORTS_DIR=reports
set CHECKPOINT_DIR=models

echo ================================================================
echo   Brain Tumor MRI: Medical-Grade Production Pipeline
echo ================================================================
echo   Enhanced with clinical neuroimaging techniques:
echo   - N4 Bias Field Correction
echo   - BET Skull Stripping (FSL-inspired)
echo   - Nyul Intensity Normalization
echo   - CLAHE Contrast Enhancement
echo ================================================================
echo.

REM ==========================================
REM Step 1: Virtual Environment Setup
REM ==========================================
echo [1/12] Checking Python virtual environment...

if exist "%VENV_DIR%" (
    echo Virtual environment found at '%VENV_DIR%'
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        echo Activating virtual environment...
        call "%VENV_DIR%\Scripts\activate.bat"
    ) else (
        echo Error: Virtual environment activation script not found!
        exit /b 1
    )
) else (
    echo Virtual environment not found. Creating new virtual environment...
    where python >nul 2>&1
    if errorlevel 1 (
        echo Error: python not found. Please install Python 3.10+.
        exit /b 1
    )
    python -m venv "%VENV_DIR%"
    call "%VENV_DIR%\Scripts\activate.bat"
    echo Virtual environment created and activated.
)

if not defined VIRTUAL_ENV (
    echo Error: Failed to activate virtual environment!
    exit /b 1
)
echo [OK] Active environment: %VIRTUAL_ENV%
echo.

REM ==========================================
REM Step 2: Install Dependencies
REM ==========================================
echo [2/12] Installing dependencies...

if exist "%REQUIREMENTS_FILE%" (
    python -m pip install --upgrade pip --quiet
    python -m pip install -r "%REQUIREMENTS_FILE%" --quiet
    python -m pip install scipy scikit-image --quiet
    echo [OK] Dependencies ready
) else (
    echo Error: %REQUIREMENTS_FILE% not found!
    exit /b 1
)
echo.

REM ==========================================
REM Step 3: Data Download
REM ==========================================
echo [3/12] Downloading and Merging Datasets...
if not exist "data\train" (
    echo Raw data not found. Running master download script...
    python tools\download_data.py --project_root .
    if errorlevel 1 exit /b 1
) else (
    echo [OK] Raw data found in 'data\train'. Skipping download.
)
echo.

REM ==========================================
REM Step 4: Medical-Grade Preprocessing
REM ==========================================
echo [4/12] Medical-Grade Preprocessing Pipeline
echo ===========================================

call :preprocess "data\train" "data\train_medical" "Training Set (Masoud + Pradeep)"
call :preprocess "data\val" "data\val_medical" "Validation Set"
call :preprocess "data\test" "data\test_medical" "Test Set"
call :preprocess "data\external_navoneel" "%EXTERNAL_DATA_MEDICAL%" "External Dataset (Navoneel)"

echo [OK] All preprocessing completed
echo.

REM ==========================================
REM Step 5: Base Model Training + Eval + Error Analysis
REM ==========================================
echo [5/12] Training Base Model (EfficientNetV2)...
if not exist "%CHECKPOINT_DIR%\best.keras" (
    python src\train.py --config %CONFIG_FILE%
    if errorlevel 1 exit /b 1
) else (
    echo [SKIP] models\best.keras already exists.
)

echo Evaluating Base Model...
python src\eval.py --config %CONFIG_FILE%
if errorlevel 1 exit /b 1
echo.

echo Running Baseline Error Analysis (non-blocking)...
python src\error_analysis.py
echo.

REM ==========================================
REM Step 6: Comparative Analysis (Optional)
REM ==========================================
echo [6/12] Comparative Analysis: Medical vs Legacy
if exist "data\train_cropped\preprocessing_summary.json" (
    if exist "data\train_medical\preprocessing_summary.json" (
        echo [INFO] Legacy vs Medical comparison available (run notebooks for full report).
    ) else (
        echo [SKIP] Medical preprocessing summary not found.
    )
) else (
    echo [SKIP] Legacy preprocessing not found. Skipping comparison.
)
echo.

REM ==========================================
REM Step 7: Fine-Tuning (Sensitivity Improvement)
REM ==========================================
echo [7/12] Fine-Tuning for External Data Adaptation...
if not exist "%CHECKPOINT_DIR%\finetuned_navoneel.keras" (
    python tools\train_finetune.py --config %CONFIG_FILE% --data %EXTERNAL_DATA_MEDICAL%
    if errorlevel 1 exit /b 1
) else (
    echo [SKIP] models\finetuned_navoneel.keras already exists.
)
echo.

REM ==========================================
REM Step 8: External Validation (Base + Fine-Tuned/Ensemble)
REM ==========================================
echo [8/12] External Validation on Navoneel Dataset...

if exist "%CHECKPOINT_DIR%\finetuned_navoneel.keras" ren "%CHECKPOINT_DIR%\finetuned_navoneel.keras" "finetuned_navoneel.keras.bak"
python tools\evaluate_external.py --config %CONFIG_FILE% --data %EXTERNAL_DATA_MEDICAL% --split full --fn-topk 12 %EVAL_TTA_ARGS% %LOG_WANDB_ARG%
if exist "%CHECKPOINT_DIR%\finetuned_navoneel.keras.bak" ren "%CHECKPOINT_DIR%\finetuned_navoneel.keras.bak" "finetuned_navoneel.keras"

python tools\evaluate_external.py --config %CONFIG_FILE% --data %EXTERNAL_DATA_MEDICAL% --split full --fn-topk 12 %EVAL_TTA_ARGS% %LOG_WANDB_ARG%
echo.

REM ==========================================
REM Step 9: Adaptive Retraining (Focal) + Optional TTA Audit
REM ==========================================
echo [9/12] Adaptive Retraining with Focal Loss...
if not exist "%CHECKPOINT_DIR%\focal_best.keras" (
    python tools\adaptive_retrain.py --config %CONFIG_FILE% --external_data %EXTERNAL_DATA_MEDICAL%
) else (
    echo [SKIP] models\focal_best.keras already exists.
)

if "%ENABLE_TTA%"=="1" (
    echo [INFO] Running optional TTA evaluation (5 samples) for focal model...
    python tools\adaptive_retrain.py --config %CONFIG_FILE% --external_data %EXTERNAL_DATA_MEDICAL% --skip_training --use_tta --n_tta 5
)
echo.

REM ==========================================
REM Step 10: Threshold Optimization
REM ==========================================
echo [10/12] Optimizing Detection Threshold...
python tools\optimize_threshold.py --config %CONFIG_FILE% --data %EXTERNAL_DATA_MEDICAL% %LOG_WANDB_ARG%
if errorlevel 1 exit /b 1
echo.

REM ==========================================
REM Step 11: Comparative Dashboards
REM ==========================================
echo [11/12] Generating Comparative Dashboards...
python tools\compare_models.py --config %CONFIG_FILE% --output %REPORTS_DIR% --checkpoint_dir %CHECKPOINT_DIR%
echo.

REM ==========================================
REM Final Summary
REM ==========================================
echo ================================================================
echo              MEDICAL-GRADE PIPELINE COMPLETED
echo ================================================================
echo Generated Models:
echo   - Base:        models\best.keras
echo   - Fine-Tuned:  models\finetuned_navoneel.keras
echo   - Focal:       models\focal_best.keras
echo External Results:
echo   - Base eval:   models\base_external_results_full.json
echo   - FT/Ens eval: models\finetuned_external_results_full.json (o ensemble_external_results_full.json)
echo   - Threshold:   reports\threshold_optimization.json
echo Reports dir:     %REPORTS_DIR%
echo Data dirs:       data\train_medical\, data\external_navoneel_medical\
echo ================================================================

goto :EOF

:preprocess
set SRC=%~1
set DST=%~2
set DESC=%~3
echo Processing: %DESC%
echo    Source:      %SRC%
echo    Destination: %DST%
if not exist "%DST%\preprocessing_summary.json" (
    python tools\preprocess_dataset.py --input_dir %SRC% --output_dir %DST% --config %CONFIG_FILE% --mode auto
    if errorlevel 1 (
        echo [FAILED] %DESC% preprocessing failed!
        exit /b 1
    )
    echo [OK] Completed: %DESC%
) else (
    echo [OK] Already processed: %DESC%
)
echo.
exit /b 0
