#!/bin/bash
# run.sh - Complete Pipeline: Data Prep, Training, Fine-Tuning, and Evaluation
# ------------------------------------------------------------------
# 1. Configure virtual environment
# 2. Install dependencies
# 3. Download & Merge ALL datasets (Unified Script)
# 4. Preprocess Images (Skull Stripping)
# 5. Train base model (EfficientNetV2 on Cropped Data)
# 6. Evaluate base model
# 7. Execute Fine-Tuning on External Data (Cropped)
# 8. Evaluate optimized model

set -e # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Add root directory to PYTHONPATH
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
echo -e "${GREEN}[1/8] Checking virtual environment...${NC}"
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
echo -e "${GREEN}[2/8] Installing dependencies...${NC}"
pip install --upgrade pip -q
pip install -r "$REQUIREMENTS_FILE" -q
echo -e "${GREEN}✓ Dependencies ready${NC}"
echo ""

# ==========================================
# 3. Unified Data Download
# ==========================================
echo -e "${GREEN}[3/8] Downloading and Merging Datasets...${NC}"
# We check if the raw 'data/train' exists. If not, we assume we need to download.
if [ ! -d "data/train" ]; then
  echo -e "${YELLOW}Raw data not found. Running master download script...${NC}"
  python tools/download_data.py --project_root .
else
  echo -e "${GREEN}✓ Raw data found in 'data/train'. Skipping download.${NC}"
fi
echo ""

# ==========================================
# 4. Data Preprocessing (Skull Stripping)
# ==========================================
echo -e "${GREEN}[4/8] Preprocessing (Skull Stripping)...${NC}"

# Helper function to preprocess a split if the destination doesn't exist
preprocess_split() {
  SRC=$1
  DST=$2
  if [ ! -d "$DST" ]; then
    echo -e "   -> Processing $SRC to $DST..."
    python tools/preprocess_dataset.py --input_dir "$SRC" --output_dir "$DST"
  else
    echo -e "   -> $DST already exists. Skipping."
  fi
}

# Preprocess Main Splits (Masoud + Pradeep)
preprocess_split "data/train" "data/train_cropped"
preprocess_split "data/val" "data/val_cropped"
preprocess_split "data/test" "data/test_cropped"

# Preprocess External Dataset (Navoneel) for consistent Fine-Tuning
preprocess_split "data/external_navoneel" "data/external_navoneel_cropped"

echo -e "${GREEN}✓ Data preprocessing completed.${NC}"
echo ""

# ==========================================
# 5. Base Model Training
# ==========================================
echo -e "${GREEN}[5/8] Training Base Model (EfficientNetV2)...${NC}"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/.venv/lib
# Ensure config points to 'data/train_cropped' before running this!

if [ ! -f "models/best.keras" ]; then
  python src/train.py --config "$CONFIG_FILE"
else
  echo -e "${YELLOW}'models/best.keras' already exists. Skipping base training.${NC}"
fi

# Base Evaluation
echo -e "${GREEN}Evaluating Base Model...${NC}"
python src/eval.py --config "$CONFIG_FILE"
echo ""

# ==========================================
# 6. Fine-Tuning (Adaptation)
# ==========================================
echo -e "${GREEN}[6/8] Executing Fine-Tuning (Sensitivity Improvement)...${NC}"
# We use the CROPPED external dataset to match the domain of the base model
EXTERNAL_DATA_CROPPED="data/external_navoneel_cropped"

if [ ! -f "models/finetuned_navoneel.keras" ]; then
  python tools/train_finetune.py --config "$CONFIG_FILE" --data "$EXTERNAL_DATA_CROPPED"
else
  echo -e "${YELLOW}Model 'models/finetuned_navoneel.keras' already exists.${NC}"
  echo -e "Skipping fine-tuning step..."
fi
echo ""

# ==========================================
# 7. Final External Evaluation
# ==========================================
echo -e "${GREEN}[7/8] Final Evaluation on External Data...${NC}"

# Evaluate the Fine-Tuned model on the External Cropped data
python tools/evaluate_external.py --config "$CONFIG_FILE" --data "$EXTERNAL_DATA_CROPPED"

echo ""
echo -e "${GREEN}Calculating optimal threshold...${NC}"
python tools/optimize_threshold.py --config "$CONFIG_FILE" --data "$EXTERNAL_DATA_CROPPED"

echo ""

# ==========================================
# Summary
# ==========================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}       PIPELINE COMPLETED 🎉           ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Generated Models:${NC}"
echo -e "  1. Base:      ${YELLOW}models/best.keras${NC} (Cropped data)"
echo -e "  2. Optimized: ${YELLOW}models/finetuned_navoneel.keras${NC} (Fine-tuned)"
echo ""
echo -e "${GREEN}Note:${NC}"
echo -e "  Used 'data/train_cropped' for training and 'data/external_navoneel_cropped' for fine-tuning."
echo ""
