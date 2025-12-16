# run.ps1 - Complete Medical-Grade Pipeline (Windows PowerShell)
# Mirrors run.sh (12 etapas) con evaluación base + ensemble/triage,
# retraining focal opcional, TTA opcional y logging W&B opcional.

$ErrorActionPreference = "Stop"

# Toggles (set $env:ENABLE_WANDB_PIPELINE=1 or $env:ENABLE_TTA=1 before running)
if (-not $env:ENABLE_WANDB_PIPELINE) { $env:ENABLE_WANDB_PIPELINE = "0" }
if (-not $env:ENABLE_TTA) { $env:ENABLE_TTA = "0" }
$LOG_WANDB_ARG = if ($env:ENABLE_WANDB_PIPELINE -eq "1") { "--log-wandb" } else { "" }
$EVAL_TTA_ARGS = if ($env:ENABLE_TTA -eq "1") { "--use-tta" } else { "" }

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $SCRIPT_DIR
$env:PYTHONPATH = "$SCRIPT_DIR;$env:PYTHONPATH"
$env:TF_ENABLE_ONEDNN_OPTS = "0"

$VENV_DIR = ".venv"
$REQUIREMENTS_FILE = "requirements.txt"
$CONFIG_FILE = "configs/config.yaml"
$EXTERNAL_DATA_MEDICAL = "data/external_navoneel_medical"
$REPORTS_DIR = "reports"
$CHECKPOINT_DIR = "models"

Write-Host "================================================================"
Write-Host "  Brain Tumor MRI: Medical-Grade Production Pipeline"
Write-Host "================================================================"
Write-Host "  Enhanced with clinical neuroimaging techniques:"
Write-Host "  - N4 Bias Field Correction"
Write-Host "  - BET Skull Stripping (FSL-inspired)"
Write-Host "  - Nyul Intensity Normalization"
Write-Host "  - CLAHE Contrast Enhancement"
Write-Host "================================================================"
Write-Host ""

# 1. Virtual environment
Write-Host "[1/12] Checking Python virtual environment..."
if (Test-Path $VENV_DIR) {
    Write-Host "Virtual environment found at '$VENV_DIR'"
    $activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
    if (-not (Test-Path $activateScript)) { throw "Activation script not found" }
    & $activateScript
} else {
    Write-Host "Creating virtual environment..."
    python -m venv $VENV_DIR
    $activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
    & $activateScript
    Write-Host "Virtual environment created and activated."
}
if (-not $env:VIRTUAL_ENV) { throw "Failed to activate virtual environment" }
Write-Host "[OK] Active environment: $env:VIRTUAL_ENV"
Write-Host ""

# 2. Dependencies
Write-Host "[2/12] Installing dependencies..."
if (-not (Test-Path $REQUIREMENTS_FILE)) { throw "$REQUIREMENTS_FILE not found" }
python -m pip install --upgrade pip --quiet
python -m pip install -r $REQUIREMENTS_FILE --quiet
python -m pip install scipy scikit-image --quiet
Write-Host "[OK] Dependencies ready"
Write-Host ""

# 3. Data download
Write-Host "[3/12] Downloading and Merging Datasets..."
if (-not (Test-Path "data\train")) {
    python tools\download_data.py --project_root .
}
Write-Host ""

# 4. Medical-grade preprocessing
Write-Host "[4/12] Medical-Grade Preprocessing Pipeline"
function Invoke-MedicalPreprocessing {
    param([string]$Source,[string]$Destination,[string]$Description)
    Write-Host "Processing: $Description"
    $summary = Join-Path $Destination "preprocessing_summary.json"
    if ((-not (Test-Path $Destination)) -or (-not (Test-Path $summary))) {
        python tools\preprocess_dataset.py --input_dir $Source --output_dir $Destination --config $CONFIG_FILE --mode auto
        if ($LASTEXITCODE -ne 0) { throw "Preprocessing failed: $Description" }
    } else {
        Write-Host "[OK] Already processed: $Description"
    }
    Write-Host ""
}
Invoke-MedicalPreprocessing "data\train" "data\train_medical" "Training Set (Masoud + Pradeep)"
Invoke-MedicalPreprocessing "data\val" "data\val_medical" "Validation Set"
Invoke-MedicalPreprocessing "data\test" "data\test_medical" "Test Set"
Invoke-MedicalPreprocessing "data\external_navoneel" $EXTERNAL_DATA_MEDICAL "External Dataset (Navoneel)"
Write-Host "[OK] All preprocessing completed"
Write-Host ""

# 5. Train base + eval + error analysis
Write-Host "[5/12] Training Base Model (EfficientNetV2)..."
if (-not (Test-Path "$CHECKPOINT_DIR\best.keras")) {
    python src\train.py --config $CONFIG_FILE
    if ($LASTEXITCODE -ne 0) { throw "Training failed" }
} else {
    Write-Host "[SKIP] models\best.keras already exists."
}
Write-Host "Evaluating Base Model..."
python src\eval.py --config $CONFIG_FILE
if ($LASTEXITCODE -ne 0) { throw "Eval failed" }
Write-Host "Running Baseline Error Analysis (non-blocking)..."
python src\error_analysis.py
Write-Host ""

# 6. Comparative Analysis (optional)
Write-Host "[6/12] Comparative Analysis: Medical vs Legacy"
if ((Test-Path "data\train_cropped\preprocessing_summary.json") -and (Test-Path "data\train_medical\preprocessing_summary.json")) {
    Write-Host "[INFO] Legacy vs Medical comparison available (see notebooks)."
} else {
    Write-Host "[SKIP] Legacy preprocessing not found."
}
Write-Host ""

# 7. Fine-Tuning
Write-Host "[7/12] Fine-Tuning for External Data Adaptation..."
if (-not (Test-Path "$CHECKPOINT_DIR\finetuned_navoneel.keras")) {
    python tools\train_finetune.py --config $CONFIG_FILE --data $EXTERNAL_DATA_MEDICAL
    if ($LASTEXITCODE -ne 0) { throw "Fine-tune failed" }
} else { Write-Host "[SKIP] finetuned_navoneel.keras already exists." }
Write-Host ""

# 8. External Validation (base + fine-tuned/ensemble)
Write-Host "[8/12] External Validation on Navoneel Dataset..."
if (Test-Path "$CHECKPOINT_DIR\finetuned_navoneel.keras") { Rename-Item "$CHECKPOINT_DIR\finetuned_navoneel.keras" "finetuned_navoneel.keras.bak" -Force }
python tools\evaluate_external.py --config $CONFIG_FILE --data $EXTERNAL_DATA_MEDICAL --split full --fn-topk 12 $EVAL_TTA_ARGS $LOG_WANDB_ARG
if (Test-Path "$CHECKPOINT_DIR\finetuned_navoneel.keras.bak") { Rename-Item "$CHECKPOINT_DIR\finetuned_navoneel.keras.bak" "finetuned_navoneel.keras" -Force }
python tools\evaluate_external.py --config $CONFIG_FILE --data $EXTERNAL_DATA_MEDICAL --split full --fn-topk 12 $EVAL_TTA_ARGS $LOG_WANDB_ARG
Write-Host ""

# 9. Adaptive Retraining (Focal) + optional TTA audit
Write-Host "[9/12] Adaptive Retraining with Focal Loss..."
if (-not (Test-Path "$CHECKPOINT_DIR\focal_best.keras")) {
    python tools\adaptive_retrain.py --config $CONFIG_FILE --external_data $EXTERNAL_DATA_MEDICAL
}
if ($env:ENABLE_TTA -eq "1") {
    Write-Host "[INFO] Running optional TTA evaluation (5 samples) for focal model..."
    python tools\adaptive_retrain.py --config $CONFIG_FILE --external_data $EXTERNAL_DATA_MEDICAL --skip_training --use_tta --n_tta 5
}
Write-Host ""

# 10. Threshold Optimization
Write-Host "[10/12] Optimizing Detection Threshold..."
python tools\optimize_threshold.py --config $CONFIG_FILE --data $EXTERNAL_DATA_MEDICAL $LOG_WANDB_ARG
if ($LASTEXITCODE -ne 0) { throw "Threshold optimization failed" }
Write-Host ""

# 11. Comparative Dashboards
Write-Host "[11/12] Generating Comparative Dashboards..."
python tools\compare_models.py --config $CONFIG_FILE --output $REPORTS_DIR --checkpoint_dir $CHECKPOINT_DIR
Write-Host ""

# Final Summary
Write-Host "================================================================"
Write-Host "              MEDICAL-GRADE PIPELINE COMPLETED"
Write-Host "================================================================"
Write-Host "Models: base (best.keras), fine-tuned (finetuned_navoneel.keras), focal (focal_best.keras)"
Write-Host "External results: base_external_results_full.json, finetuned/ensemble_external_results_full.json, threshold_optimization.json"
Write-Host "Reports: $REPORTS_DIR"
Write-Host "Data: data\train_medical\, data\external_navoneel_medical\"
Write-Host "================================================================"
