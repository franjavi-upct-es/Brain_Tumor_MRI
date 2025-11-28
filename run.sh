#!/bin/bash
# run.sh - Complete pipeline execution script for Brain Tumor MRI Classification
# This script:
# 1. Detects or creates a Python virtual environment
# 2. Installs required dependencies
# 3. Executes the training pipeline
# 4. Runs evaluation and generates all reports/figures
# 5. Displays a summary of results

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Virtual environment settings
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"
CONFIG_FILE="configs/config.yaml"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Brain Tumor MRI Classification Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ==========================================
# Step 1: Virtual Environment Setup
# ==========================================
echo -e "${GREEN}[1/5] Checking Python virtual environment...${NC}"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment found at '$VENV_DIR'${NC}"
    
    # Activate existing virtual environment
    if [ -f "$VENV_DIR/bin/activate" ]; then
        echo -e "${GREEN}Activating virtual environment...${NC}"
        source "$VENV_DIR/bin/activate"
    else
        echo -e "${RED}Error: Virtual environment activation script not found!${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Virtual environment not found. Creating new virtual environment...${NC}"
    
    # Check if python3 is available
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: python3 not found. Please install Python 3.${NC}"
        exit 1
    fi
    
    # Create virtual environment
    python3 -m venv "$VENV_DIR"
    
    # Activate the new virtual environment
    source "$VENV_DIR/bin/activate"
    
    echo -e "${GREEN}Virtual environment created and activated.${NC}"
fi

# Verify we're in the virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}Error: Failed to activate virtual environment!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Virtual environment active: $VIRTUAL_ENV${NC}"
echo ""

# ==========================================
# Step 2: Install Dependencies
# ==========================================
echo -e "${GREEN}[2/5] Installing dependencies...${NC}"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${YELLOW}Installing packages from $REQUIREMENTS_FILE...${NC}"
    pip install --upgrade pip -q
    pip install -r "$REQUIREMENTS_FILE" -q
    echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
else
    echo -e "${RED}Error: $REQUIREMENTS_FILE not found!${NC}"
    exit 1
fi
echo ""

# ==========================================
# Step 3: Verify Configuration
# ==========================================
echo -e "${GREEN}[3/5] Verifying configuration...${NC}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file '$CONFIG_FILE' not found!${NC}"
    exit 1
fi

# Check if data directory exists
DATA_DIR="data"
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${YELLOW}Warning: Data directory '$DATA_DIR' not found!${NC}"
    echo -e "${YELLOW}You may need to run: python tools/download_and_prepare_kaggle.py${NC}"
    echo -e "${YELLOW}Do you want to continue anyway? (y/n)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborting pipeline.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Configuration verified${NC}"
echo ""

# ==========================================
# Step 4: Training
# ==========================================
echo -e "${GREEN}[4/5] Starting training pipeline...${NC}"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/.venv/lib
echo -e "${YELLOW}Configurada ruta de librerías GPU: .venv/lib${NC}"
echo ""

if [ -f "src/train.py" ]; then
    python src/train.py --config "$CONFIG_FILE"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Training completed successfully${NC}"
    else
        echo -e "${RED}✗ Training failed!${NC}"
        exit 1
    fi
else
    echo -e "${RED}Error: src/train.py not found!${NC}"
    exit 1
fi
echo ""

# ==========================================
# Step 5: Evaluation and Figure Generation
# ==========================================
echo -e "${GREEN}[5/5] Running evaluation and generating reports...${NC}"

if [ -f "src/eval.py" ]; then
    python src/eval.py --config "$CONFIG_FILE"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Evaluation completed successfully${NC}"
    else
        echo -e "${RED}✗ Evaluation failed!${NC}"
        exit 1
    fi
else
    echo -e "${RED}Error: src/eval.py not found!${NC}"
    exit 1
fi
echo ""

# ==========================================
# Summary
# ==========================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Pipeline Execution Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Results and artifacts:${NC}"
echo -e "  • Model checkpoint: ${YELLOW}models/best.keras${NC}"
echo -e "  • Training history: ${YELLOW}training_log.csv${NC}"
echo -e "  • Calibration params: ${YELLOW}models/temperature.json${NC}"
echo ""
echo -e "${GREEN}Generated figures and reports in 'reports/':${NC}"
echo -e "  • Training curves: ${YELLOW}acc_curve.png, loss_curve.png${NC}"
echo -e "  • Confusion matrices: ${YELLOW}cm.png, cm_norm.png${NC}"
echo -e "  • ROC/PR curves: ${YELLOW}roc_curves.png, pr_curves.png${NC}"
echo -e "  • Calibration: ${YELLOW}reliability_diagram.png, confidence_hist.png${NC}"
echo -e "  • Classification report: ${YELLOW}classification_report.txt${NC}"
echo -e "  • Metrics summary: ${YELLOW}summary.json, calibration_metrics.json${NC}"
echo ""
echo -e "${GREEN}Grad-CAM visualizations (if generated):${NC}"
echo -e "  • Location: ${YELLOW}gradcam_samples/${NC}"
echo ""
echo -e "${BLUE}TensorBoard logs available in:${NC}"
echo -e "  • ${YELLOW}tb/${NC}"
echo -e "  • Run: ${YELLOW}tensorboard --logdir=tb/${NC}"
echo ""
echo -e "${GREEN}To run inference on a single image:${NC}"
echo -e "  ${YELLOW}python src/infer.py --config $CONFIG_FILE --image <path_to_image>${NC}"
echo ""
echo -e "${GREEN}To run k-fold cross-validation:${NC}"
echo -e "  ${YELLOW}python src/train_kfold.py --config $CONFIG_FILE${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All done! 🎉${NC}"
echo -e "${BLUE}========================================${NC}"
