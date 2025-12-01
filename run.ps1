# run.ps1 - Complete pipeline execution script for Brain Tumor MRI Classification (Windows)
# This script:
# 1. Detects or creates a Python virtual environment
# 2. Installs required dependencies
# 3. Executes the training pipeline
# 4. Runs evaluation and generates all reports/figures
# 5. Displays a summary of results

# Stop on errors
$ErrorActionPreference = "Stop"

# Get script directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $SCRIPT_DIR

# Add project root to PYTHONPATH so Python can find the 'src' module
$env:PYTHONPATH = "$SCRIPT_DIR;$env:PYTHONPATH"

# Virtual environment settings
$VENV_DIR = ".venv"
$REQUIREMENTS_FILE = "requirements.txt"
$CONFIG_FILE = "configs/config.yaml"

# Color functions for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Green { Write-ColorOutput Green $args }
function Write-Blue { Write-ColorOutput Cyan $args }
function Write-Yellow { Write-ColorOutput Yellow $args }
function Write-Red { Write-ColorOutput Red $args }

Write-Blue "========================================"
Write-Blue "  Brain Tumor MRI Classification Pipeline"
Write-Blue "========================================"
Write-Output ""

# ==========================================
# Step 1: Virtual Environment Setup
# ==========================================
Write-Green "[1/5] Checking Python virtual environment..."

if (Test-Path $VENV_DIR) {
    Write-Yellow "Virtual environment found at '$VENV_DIR'"
    
    # Activate existing virtual environment
    $activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        Write-Green "Activating virtual environment..."
        & $activateScript
    } else {
        Write-Red "Error: Virtual environment activation script not found!"
        exit 1
    }
} else {
    Write-Yellow "Virtual environment not found. Creating new virtual environment..."
    
    # Check if python is available
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        Write-Red "Error: python not found. Please install Python 3."
        exit 1
    }
    
    # Create virtual environment
    python -m venv $VENV_DIR
    
    # Activate the new virtual environment
    $activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
    & $activateScript
    
    Write-Green "Virtual environment created and activated."
}

# Verify we're in the virtual environment
if (-not $env:VIRTUAL_ENV) {
    Write-Red "Error: Failed to activate virtual environment!"
    exit 1
}

Write-Green "✓ Virtual environment active: $env:VIRTUAL_ENV"
Write-Output ""

# ==========================================
# Step 2: Install Dependencies
# ==========================================
Write-Green "[2/5] Installing dependencies..."

if (Test-Path $REQUIREMENTS_FILE) {
    Write-Yellow "Installing packages from $REQUIREMENTS_FILE..."
    python -m pip install --upgrade pip --quiet
    python -m pip install -r $REQUIREMENTS_FILE --quiet
    Write-Green "✓ Dependencies installed successfully"
} else {
    Write-Red "Error: $REQUIREMENTS_FILE not found!"
    exit 1
}
Write-Output ""

# ==========================================
# Step 3: Verify Configuration
# ==========================================
Write-Green "[3/5] Verifying configuration..."

if (-not (Test-Path $CONFIG_FILE)) {
    Write-Red "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
}

# Check if data directory exists
$DATA_DIR = "data"
if (-not (Test-Path $DATA_DIR)) {
    Write-Yellow "Warning: Data directory '$DATA_DIR' not found!"
    Write-Yellow "You may need to run: python tools\download_and_prepare_kaggle.py"
    Write-Yellow "Do you want to continue anyway? (y/n)"
    $response = Read-Host
    if ($response -notmatch '^[Yy]$') {
        Write-Red "Aborting pipeline."
        exit 1
    }
}

Write-Green "✓ Configuration verified"
Write-Output ""

# ==========================================
# Step 4: Training
# ==========================================
Write-Green "[4/5] Starting training pipeline..."
Write-Blue "This may take a while depending on your hardware."
Write-Output ""

if (Test-Path "models\best.keras") {
    Write-Green "✓ Model 'models\best.keras' already exists. Skipping base training."
    Write-Yellow "   (Delete 'models' folder to retrain from scratch)"
} else {
    if (Test-Path "src\train.py") {
        python src\train.py --config $CONFIG_FILE
        
        if ($LASTEXITCODE -eq 0) {
            Write-Green "✓ Training completed successfully"
        } else {
            Write-Red "✗ Training failed!"
            exit 1
        }
    } else {
        Write-Red "Error: src\train.py not found!"
        exit 1
    }
}
Write-Output ""

# ==========================================
# Step 5: Evaluation and Figure Generation
# ==========================================
Write-Green "[5/5] Running evaluation and generating reports..."

if (Test-Path "src\eval.py") {
    python src\eval.py --config $CONFIG_FILE
    
    if ($LASTEXITCODE -eq 0) {
        Write-Green "✓ Evaluation completed successfully"
    } else {
        Write-Red "✗ Evaluation failed!"
        exit 1
    }
} else {
    Write-Red "Error: src\eval.py not found!"
    exit 1
}
Write-Output ""

# ==========================================
# Summary
# ==========================================
Write-Blue "========================================"
Write-Blue "  Pipeline Execution Complete!"
Write-Blue "========================================"
Write-Output ""
Write-Green "Results and artifacts:"
Write-Output "  • Model checkpoint: " -NoNewline; Write-Yellow "models\best.keras"
Write-Output "  • Training history: " -NoNewline; Write-Yellow "training_log.csv"
Write-Output "  • Calibration params: " -NoNewline; Write-Yellow "models\temperature.json"
Write-Output ""
Write-Green "Generated figures and reports in 'reports\':"
Write-Output "  • Training curves: " -NoNewline; Write-Yellow "acc_curve.png, loss_curve.png"
Write-Output "  • Confusion matrices: " -NoNewline; Write-Yellow "cm.png, cm_norm.png"
Write-Output "  • ROC/PR curves: " -NoNewline; Write-Yellow "roc_curves.png, pr_curves.png"
Write-Output "  • Calibration: " -NoNewline; Write-Yellow "reliability_diagram.png, confidence_hist.png"
Write-Output "  • Classification report: " -NoNewline; Write-Yellow "classification_report.txt"
Write-Output "  • Metrics summary: " -NoNewline; Write-Yellow "summary.json, calibration_metrics.json"
Write-Output ""
Write-Green "Grad-CAM visualizations (if generated):"
Write-Output "  • Location: " -NoNewline; Write-Yellow "gradcam_samples\"
Write-Output ""
Write-Blue "TensorBoard logs available in:"
Write-Output "  • " -NoNewline; Write-Yellow "tb\"
Write-Output "  • Run: " -NoNewline; Write-Yellow "tensorboard --logdir=tb\"
Write-Output ""
Write-Green "To run inference on a single image:"
Write-Output "  " -NoNewline; Write-Yellow "python src\infer.py --config $CONFIG_FILE --image <path_to_image>"
Write-Output ""
Write-Green "To run k-fold cross-validation:"
Write-Output "  " -NoNewline; Write-Yellow "python src\train_kfold.py --config $CONFIG_FILE"
Write-Output ""
Write-Blue "========================================"
Write-Green "All done! 🎉"
Write-Blue "========================================"
