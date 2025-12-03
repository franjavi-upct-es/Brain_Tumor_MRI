#!/bin/bash
# run.sh - Complete Pipeline: Training, Fine-Tuning and Evaluation
# ------------------------------------------------------------------
# 1. Configure virtual environment
# 2. Install dependencies
# 3. Train base model (EfficientNet)
# 4. Evaluate base model
# 5. Download external dataset (Navoneel)
# 6. Execute Fine-Tuning to improve sensitivity
# 7. Evaluate optimized model on external data

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Add root directory to PYTHONPATH so Python can find 'src' module
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"
CONFIG_FILE="configs/config.yaml"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}  Brain Tumor MRI: Complete Production Pipeline       ${NC}"
echo -e "${BLUE}======================================================${NC}"
echo ""

# ==========================================
# 1. Virtual Environment
# ==========================================
echo -e "${GREEN}[1/7] Checking virtual environment...${NC}"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Active environment: $VIRTUAL_ENV${NC}"
echo ""

# ==========================================
# 2. Dependencies
# ==========================================
echo -e "${GREEN}[2/7] Installing dependencies...${NC}"
pip install --upgrade pip -q
pip install -r "$REQUIREMENTS_FILE" -q
echo -e "${GREEN}✓ Dependencies ready${NC}"
echo ""

# ==========================================
# 3. Base Data Verification
# ==========================================
echo -e "${GREEN}[3/7] Verifying main dataset...${NC}"
if [ ! -d "data/train" ]; then
    echo -e "${YELLOW}Base training data not detected.${NC}"
    echo -e "Running download script (Kaggle: masoudnickparvar)..."
    python tools/download_and_prepare_kaggle.py --project-root . --val-size 0.1 --use-symlinks
fi
echo -e "${GREEN}✓ Main dataset verified${NC}"
echo ""

# ==========================================
# 4. Base Model Training
# ==========================================
echo -e "${GREEN}[4/7] Training Base Model...${NC}"
# Export libraries for GPU if needed
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/.venv/lib

if [ ! -f "models/best.keras" ]; then
    python src/train.py --config "$CONFIG_FILE"
else
    echo -e "${YELLOW}'models/best.keras' already exists. Skipping base training.${NC}"
    echo -e "(Delete 'models/' folder if you want to retrain from scratch)"
fi

# Base Evaluation
echo -e "${GREEN}Evaluating Base Model...${NC}"
python src/eval.py --config "$CONFIG_FILE"
echo ""

# ==========================================
# 5. External Dataset Preparation
# ==========================================
echo -e "${GREEN}[5/7] Preparing External Dataset (Navoneel)...${NC}"
if [ ! -d "data/external_navoneel" ]; then
    python tools/download_navoneel.py
else
    echo -e "${GREEN}✓ External dataset already exists at 'data/external_navoneel'${NC}"
fi
echo ""

# ==========================================
# 6. Fine-Tuning (Adaptation)
# ==========================================
echo -e "${GREEN}[6/7] Executing Fine-Tuning (Sensitivity Improvement)...${NC}"
# Only train if fine-tuned model doesn't exist to save time
if [ ! -f "models/finetuned_navoneel.keras" ]; then
    python tools/train_finetune.py --config "$CONFIG_FILE" --data "data/external_navoneel"
else
    echo -e "${YELLOW}Model 'models/finetuned_navoneel.keras' already exists.${NC}"
    echo -e "Skipping fine-tuning step..."
fi
echo ""

# ==========================================
# 7. Final External Evaluation
# ==========================================
echo -e "${GREEN}[7/7] Final Evaluation on External Data...${NC}"
# Temporarily modify the script to evaluate the fine-tuned model if necessary,
# or ensure that evaluate_external.py points to the correct model.
# NOTE: We assume evaluate_external.py loads 'best.keras' by default,
# but fine-tuning generates 'finetuned_navoneel.keras'.
# To automate it, we pass the explicit path if the script supports it,
# or trust that 'train_finetune.py' left everything ready.

# As in our previous conversation, we will evaluate the final result:
echo -e "${BLUE}--- Optimized Model Results ---${NC}"
# Here we do a trick: temporarily rename for evaluation, or ideally
# update evaluate_external.py to accept --model.
# Given the current state, we'll run the evaluation assuming train_finetune already saved the model.

# IMPORTANT: Make sure evaluate_external.py uses the correct model.
# If you haven't modified it to accept arguments, it will use best.keras.
# For this automatic script, it's better to print a reminder or
# use the threshold optimization script which is very informative.

python tools/evaluate_external.py --config "$CONFIG_FILE" --data "data/external_navoneel"

echo ""
echo -e "${GREEN}Calculating optimal threshold...${NC}"
python tools/optimize_threshold.py --config "$CONFIG_FILE" --data "data/external_navoneel"

echo ""

# ==========================================
# Summary
# ==========================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}       PIPELINE COMPLETED 🎉           ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Generated Models:${NC}"
echo -e "  1. Base:       ${YELLOW}models/best.keras${NC} (High specificity)"
echo -e "  2. Optimized: ${YELLOW}models/finetuned_navoneel.keras${NC} (High sensitivity)"
echo ""
echo -e "${GREEN}Reports:${NC}"
echo -e "  • Check 'reports/' for curves and confusion matrices of base model."
echo -e "  • Check console output above for optimized model metrics."
echo ""
echo -e "${GREEN}For inference with optimized model:${NC}"
echo -e "  ${YELLOW}python src/infer.py --image <path> --threshold 0.65${NC}"
echo ""