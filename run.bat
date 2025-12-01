@echo off
REM run.bat - Complete pipeline execution script for Brain Tumor MRI Classification (Windows)
REM This script:
REM 1. Detects or creates a Python virtual environment
REM 2. Installs required dependencies
REM 3. Executes the training pipeline
REM 4. Runs evaluation and generates all reports/figures
REM 5. Displays a summary of results

setlocal enabledelayedexpansion

REM Get script directory
cd /d "%~dp0"

REM Add project root to PYTHONPATH so Python can find the 'src' module
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Virtual environment settings
set VENV_DIR=.venv
set REQUIREMENTS_FILE=requirements.txt
set CONFIG_FILE=configs\config.yaml

echo ========================================
echo   Brain Tumor MRI Classification Pipeline
echo ========================================
echo.

REM ==========================================
REM Step 1: Virtual Environment Setup
REM ==========================================
echo [1/5] Checking Python virtual environment...

if exist "%VENV_DIR%" (
    echo Virtual environment found at '%VENV_DIR%'
    
    REM Activate existing virtual environment
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        echo Activating virtual environment...
        call "%VENV_DIR%\Scripts\activate.bat"
    ) else (
        echo Error: Virtual environment activation script not found!
        exit /b 1
    )
) else (
    echo Virtual environment not found. Creating new virtual environment...
    
    REM Check if python is available
    where python >nul 2>&1
    if errorlevel 1 (
        echo Error: python not found. Please install Python 3.
        exit /b 1
    )
    
    REM Create virtual environment
    python -m venv "%VENV_DIR%"
    
    REM Activate the new virtual environment
    call "%VENV_DIR%\Scripts\activate.bat"
    
    echo Virtual environment created and activated.
)

REM Verify we're in the virtual environment
if not defined VIRTUAL_ENV (
    echo Error: Failed to activate virtual environment!
    exit /b 1
)

echo [OK] Virtual environment active: %VIRTUAL_ENV%
echo.

REM ==========================================
REM Step 2: Install Dependencies
REM ==========================================
echo [2/5] Installing dependencies...

if exist "%REQUIREMENTS_FILE%" (
    echo Installing packages from %REQUIREMENTS_FILE%...
    python -m pip install --upgrade pip --quiet
    python -m pip install -r "%REQUIREMENTS_FILE%" --quiet
    echo [OK] Dependencies installed successfully
) else (
    echo Error: %REQUIREMENTS_FILE% not found!
    exit /b 1
)
echo.

REM ==========================================
REM Step 3: Verify Configuration
REM ==========================================
echo [3/5] Verifying configuration...

if not exist "%CONFIG_FILE%" (
    echo Error: Configuration file '%CONFIG_FILE%' not found!
    exit /b 1
)

REM Check if data directory exists
set DATA_DIR=data
if not exist "%DATA_DIR%" (
    echo Warning: Data directory '%DATA_DIR%' not found!
    echo You may need to run: python tools\download_and_prepare_kaggle.py
    echo Do you want to continue anyway? (y/n)
    set /p response=
    if /i not "!response!"=="y" (
        echo Aborting pipeline.
        exit /b 1
    )
)

echo [OK] Configuration verified
echo.

REM ==========================================
REM Step 4: Training
REM ==========================================
echo [4/5] Starting training pipeline...
echo This may take a while depending on your hardware.
echo.

if exist "models\best.keras" (
    echo [OK] Model 'models\best.keras' already exists. Skipping base training.
    echo      (Delete 'models' folder to retrain from scratch)
) else (
    if exist "src\train.py" (
        python src\train.py --config "%CONFIG_FILE%"
        
        if errorlevel 1 (
            echo [FAILED] Training failed!
            exit /b 1
        ) else (
            echo [OK] Training completed successfully
        )
    ) else (
        echo Error: src\train.py not found!
        exit /b 1
    )
)
echo.

REM ==========================================
REM Step 5: Evaluation and Figure Generation
REM ==========================================
echo [5/5] Running evaluation and generating reports...

if exist "src\eval.py" (
    python src\eval.py --config "%CONFIG_FILE%"
    
    if errorlevel 1 (
        echo [FAILED] Evaluation failed!
        exit /b 1
    ) else (
        echo [OK] Evaluation completed successfully
    )
) else (
    echo Error: src\eval.py not found!
    exit /b 1
)
echo.

REM ==========================================
REM Summary
REM ==========================================
echo ========================================
echo   Pipeline Execution Complete!
echo ========================================
echo.
echo Results and artifacts:
echo   * Model checkpoint: models\best.keras
echo   * Training history: training_log.csv
echo   * Calibration params: models\temperature.json
echo.
echo Generated figures and reports in 'reports\':
echo   * Training curves: acc_curve.png, loss_curve.png
echo   * Confusion matrices: cm.png, cm_norm.png
echo   * ROC/PR curves: roc_curves.png, pr_curves.png
echo   * Calibration: reliability_diagram.png, confidence_hist.png
echo   * Classification report: classification_report.txt
echo   * Metrics summary: summary.json, calibration_metrics.json
echo.
echo Grad-CAM visualizations (if generated):
echo   * Location: gradcam_samples\
echo.
echo TensorBoard logs available in:
echo   * tb\
echo   * Run: tensorboard --logdir=tb\
echo.
echo To run inference on a single image:
echo   python src\infer.py --config %CONFIG_FILE% --image ^<path_to_image^>
echo.
echo To run k-fold cross-validation:
echo   python src\train_kfold.py --config %CONFIG_FILE%
echo.
echo ========================================
echo All done!
echo ========================================

endlocal
