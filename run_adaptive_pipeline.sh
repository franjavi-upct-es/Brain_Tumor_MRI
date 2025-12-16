#!/bin/bash
# run_adaptive_pipeline.sh - Complete Adaptive Retraining Pipeline
# ==================================================================
# Extends the base medical-grade pipeline with:
# - Baseline error analysis
# - Focal Loss retraining
# - Comparative visualization
# - Performance benchmarking

set -e # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Add root directory to PYTHONPATH so `src` and `tools` are importable
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 1. Dynamically link EACH 'lib' folder inside 'nvidia/' in site-packages
# This finds: cudnn, cublas, cuda_runtime, cufft, curand, cusolver, cusparse, nccl, etc.
export LD_LIBRARY_PATH=$(python -c 'import os, site; P=site.getsitepackages()[0]+"/nvidia"; print(":".join([os.path.join(P, d, "lib") for d in os.listdir(P) if os.path.isdir(os.path.join(P, d, "lib"))]))'):$LD_LIBRARY_PATH

# 2. Dynamically link EACH 'bin' folder (for ptxas, nvlink, etc.)
export PATH=$(python -c 'import os, site; P=site.getsitepackages()[0]+"/nvidia"; print(":".join([os.path.join(P, d, "bin") for d in os.listdir(P) if os.path.isdir(os.path.join(P, d, "bin"))]))'):$PATH

# 3. Disable oneDNN logs
export TF_ENABLE_ONEDNN_OPTS=0

echo "  -> NVIDIA libraries and binaries linked to the environment."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/.venv/lib

CONFIG_FILE="configs/config.yaml"
EXTERNAL_DATA="data/external_navoneel_medical"
CHECKPOINT_DIR="models"
REPORTS_DIR="reports"

echo -e "${MAGENTA}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                                                              ║${NC}"
echo -e "${MAGENTA}║  🧠  ADAPTIVE RETRAINING PIPELINE - ERROR CORRECTION  🎯     ║${NC}"
echo -e "${MAGENTA}║                                                              ║${NC}"
echo -e "${MAGENTA}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Goal:${NC} Transform base model from 49% → 88%+ tumor detection"
echo -e "${CYAN}Method:${NC} Focal Loss (γ=2.5) + Label Smoothing + TTA"
echo ""

# ==========================================
# Prerequisite Checks
# ==========================================
echo -e "${BLUE}[0/7] Checking Prerequisites...${NC}"

# Check if base model exists
if [ ! -f "${CHECKPOINT_DIR}/best.keras" ]; then
  echo -e "${RED}[ERROR] Base model not found: ${CHECKPOINT_DIR}/best.keras${NC}"
  echo -e "${YELLOW}Please run the base training pipeline first:${NC}"
  echo -e "  ./run.sh"
  exit 1
fi
echo -e "${GREEN}✓ Base model found${NC}"

# Check if external data is preprocessed
if [ ! -d "${EXTERNAL_DATA}" ]; then
  echo -e "${RED}[ERROR] External dataset not found: ${EXTERNAL_DATA}${NC}"
  echo -e "${YELLOW}Please run preprocessing first:${NC}"
  echo -e "  python tools/preprocess_dataset.py --input_dir data/external_navoneel --output_dir ${EXTERNAL_DATA}"
  exit 1
fi
echo -e "${GREEN}✓ External dataset found${NC}"

# Check if required Python files exist
for file in "src/losses.py" "tools/adaptive_retrain.py" "tools/compare_models.py"; do
  if [ ! -f "$file" ]; then
    echo -e "${RED}[ERROR] Required file not found: ${file}${NC}"
    echo -e "${YELLOW}Please ensure all artifacts are copied to the project${NC}"
    exit 1
  fi
done
echo -e "${GREEN}✓ All required Python modules found${NC}"
echo ""

# ==========================================
# 1. Baseline Error Analysis
# ==========================================
echo -e "${GREEN}[1/7] Baseline Error Analysis${NC}"
echo -e "${CYAN}═════════════════════════════════════════════════════════${NC}"

if [ ! -f "${REPORTS_DIR}/error_summary.json" ]; then
  echo -e "${YELLOW}Running baseline error analysis...${NC}"
  echo -e "  This analyzes systematic patterns in model failures"
  echo ""

  python src/error_analysis.py \
    --config "${CONFIG_FILE}" \
    --output_dir "${REPORTS_DIR}"

else
  echo -e "${GREEN}✓ Baseline error analysis already exists${NC}"
fi
echo ""

# ==========================================
# 2. Evaluate Base Model on External Data
# ==========================================
echo -e "${GREEN}[2/7] Evaluate Base Model (External Dataset)${NC}"
echo -e "${CYAN}═════════════════════════════════════════════════════════${NC}"

if [ ! -f "${CHECKPOINT_DIR}/base_external_results.json" ]; then
  echo -e "${YELLOW}Evaluating base model on Navoneel dataset...${NC}"
  echo -e "  This establishes the baseline performance"
  echo ""

  # Temporarily use base model for evaluation
  if [ -f "${CHECKPOINT_DIR}/finetuned_navoneel.keras" ]; then
    mv "${CHECKPOINT_DIR}/finetuned_navoneel.keras" "${CHECKPOINT_DIR}/finetuned_navoneel.keras.backup"
  fi

  python tools/evaluate_external.py \
    --config "${CONFIG_FILE}" \
    --data "${EXTERNAL_DATA}"

  # Restore fine-tuned model if it existed
  if [ -f "${CHECKPOINT_DIR}/finetuned_navoneel.keras.backup" ]; then
    mv "${CHECKPOINT_DIR}/finetuned_navoneel.keras.backup" "${CHECKPOINT_DIR}/finetuned_navoneel.keras"
  fi

  echo -e "${GREEN}✓ Base model evaluation complete${NC}"
else
  echo -e "${GREEN}✓ Base model results already exist${NC}"
fi
echo ""

# ==========================================
# 3. Adaptive Retraining with Focal Loss
# ==========================================
echo -e "${GREEN}[3/7] Adaptive Retraining (Focal Loss)${NC}"
echo -e "${CYAN}═════════════════════════════════════════════════════════${NC}"

# Check if focal model already exists
if [ -f "${CHECKPOINT_DIR}/focal_best.keras" ]; then
  echo -e "${YELLOW}Focal Loss model already exists. Overwrite? (y/N)${NC}"
  read -r -t 10 response || response="n"
  if [[ "$response" =~ ^[Yy]$ ]]; then
    SKIP_TRAINING=""
    echo -e "${YELLOW}Retraining from scratch...${NC}"
  else
    SKIP_TRAINING="--skip_training"
    echo -e "${GREEN}✓ Using existing Focal Loss model${NC}"
  fi
else
  SKIP_TRAINING=""
fi

if [ -z "$SKIP_TRAINING" ]; then
  echo -e "${YELLOW}Starting adaptive retraining pipeline...${NC}"
  echo ""
  echo -e "${CYAN}Configuration:${NC}"
  echo -e "  • Loss Function:      Focal Loss (γ=2.5, α=0.75)"
  echo -e "  • Regularization:     Label Smoothing (ε=0.1)"
  echo -e "  • Augmentation:       Tumor-focused (aggressive)"
  echo -e "  • Learning Rate:      1e-4 (fine-tuning)"
  echo -e "  • Epochs:             20"
  echo -e "  • Calibration:        Temperature scaling"
  echo ""

  python tools/adaptive_retrain.py \
    --config "${CONFIG_FILE}" \
    --external_data "${EXTERNAL_DATA}" \
    ${SKIP_TRAINING}

  echo -e "${GREEN}✓ Adaptive retraining complete${NC}"
fi
echo ""

# ==========================================
# 4. Test Time Augmentation (Optional)
# ==========================================
echo -e "${GREEN}[4/7] Test Time Augmentation (Optional)${NC}"
echo -e "${CYAN}═════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Enable TTA for maximum robustness? (y/N)${NC}"
echo -e "  ${CYAN}Note: TTA takes 5x longer but improves accuracy by ~2-3%${NC}"
read -r -t 10 response || response="n"

if [[ "$response" =~ ^[Yy]$ ]]; then
  echo -e "${YELLOW}Running TTA evaluation (5 augmentations)...${NC}"

  python tools/adaptive_retrain.py \
    --config "${CONFIG_FILE}" \
    --external_data "${EXTERNAL_DATA}" \
    --skip_training \
    --use_tta \
    --n_tta 5

  echo -e "${GREEN}✓ TTA evaluation complete${NC}"
else
  echo -e "${CYAN}Skipping TTA (using standard inference)${NC}"
fi
echo ""

# ==========================================
# 5. Comparative Analysis
# ==========================================
echo -e "${GREEN}[5/7] Generating Comparative Analysis${NC}"
echo -e "${CYAN}═════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}Creating visual comparison dashboard...${NC}"
python tools/compare_models.py \
  --config "${CONFIG_FILE}" \
  --output "${REPORTS_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}"

echo -e "${GREEN}✓ Comparative analysis complete${NC}"
echo ""

# ==========================================
# 6. Threshold Optimization
# ==========================================
echo -e "${GREEN}[6/7] Threshold Optimization${NC}"
echo -e "${CYAN}═════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}Optimizing detection threshold for clinical use...${NC}"
echo -e "  ${CYAN}Finding optimal balance between sensitivity/specificity${NC}"
echo ""

python tools/optimize_threshold.py \
  --config "${CONFIG_FILE}" \
  --data "${EXTERNAL_DATA}"

echo -e "${GREEN}✓ Threshold optimization complete${NC}"
echo ""

# ==========================================
# 7. Generate Final Report
# ==========================================
echo -e "${GREEN}[7/7] Generating Final Summary Report${NC}"
echo -e "${CYAN}═════════════════════════════════════════════════════════${NC}"

# Create comprehensive summary
python3 <<'EOF'
import json
import os
from pathlib import Path

def generate_summary():
    reports_dir = Path("reports")
    models_dir = Path("models")
    
    print("\n" + "=" * 70)
    print("ADAPTIVE RETRAINING PIPELINE - FINAL REPORT")
    print("=" * 70)
    
    # Load results
    results = {}
    for model_type in ['base', 'finetuned', 'focal']:
        json_path = models_dir / f"{model_type}_external_results.json"
        if json_path.exists():
            with open(json_path) as f:
                results[model_type] = json.load(f)
    
    if not results:
        print("\n⚠️  No results found. Pipeline may not have completed successfully.")
        return
    
    # Comparison table
    print("\n📊 PERFORMANCE COMPARISON (External Dataset)")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Base':<12} | {'Fine-Tuned':<12} | {'Focal Loss':<12}")
    print("-" * 70)
    
    metrics = ['recall', 'specificity', 'precision', 'accuracy']
    metric_names = ['Sensitivity (Recall)', 'Specificity', 'Precision (PPV)', 'Accuracy']
    
    for metric, name in zip(metrics, metric_names):
        base_val = results.get('base', {}).get(metric, 0)
        ft_val = results.get('finetuned', {}).get(metric, 0)
        focal_val = results.get('focal', {}).get(metric, 0)
        
        print(f"{name:<25} | {base_val:>11.1%} | {ft_val:>11.1%} | {focal_val:>11.1%}")
    
    # False Negatives comparison
    print("\n🎯 FALSE NEGATIVE REDUCTION (Missed Tumors)")
    print("=" * 70)
    
    for model_type, label in [('base', 'Base Model'), ('finetuned', 'Fine-Tuned'), ('focal', 'Focal Loss')]:
        if model_type in results:
            fn = results[model_type].get('false_negatives', 0)
            total = results[model_type].get('false_negatives', 0) + results[model_type].get('true_positives', 0)
            fn_rate = fn / total if total > 0 else 0
            
            status = "✅" if fn_rate < 0.15 else "⚠️"
            print(f"{label:<15} → {fn:>2} missed tumors ({fn_rate:>5.1%}) {status}")
    
    # Clinical assessment
    if 'focal' in results:
        focal_recall = results['focal'].get('recall', 0)
        print("\n🏥 CLINICAL ASSESSMENT")
        print("=" * 70)
        
        if focal_recall >= 0.85:
            print("✅ CLINICAL TARGET MET: Sensitivity ≥ 85%")
            print(f"   Current: {focal_recall:.1%}")
            print("\n   Model is ready for clinical validation:")
            print("   1. Validate on additional external datasets")
            print("   2. Conduct expert review with radiologists")
            print("   3. Prepare regulatory documentation")
        else:
            print(f"⚠️  Clinical target not yet met: {focal_recall:.1%} (need ≥85%)")
            print(f"\n   Recommended next steps:")
            print(f"   1. Increase Focal Loss gamma to 3.0")
            print(f"   2. Enable Test Time Augmentation (10 samples)")
            print(f"   3. Add more tumor samples via synthesis")
    
    # Generated artifacts
    print("\n📁 GENERATED ARTIFACTS")
    print("=" * 70)
    print(f"Models:")
    print(f"  • Base Model:          models/best.keras")
    print(f"  • Fine-Tuned Model:    models/finetuned_navoneel.keras")
    print(f"  • Focal Loss Model:    models/focal_best.keras")
    print(f"\nReports:")
    print(f"  • Comparison Dashboard: reports/model_comparison_dashboard.png")
    print(f"  • Improvement Report:   reports/improvement_report.md")
    print(f"  • Metrics Table:        reports/metrics_comparison.csv")
    print(f"  • Error Analysis:       reports/error_comparison_dashboard.png")
    
    print("\n" + "=" * 70)

generate_summary()
EOF

echo ""

# ==========================================
# Final Instructions
# ==========================================
echo -e "${MAGENTA}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                                                              ║${NC}"
echo -e "${MAGENTA}║           ADAPTIVE PIPELINE COMPLETED SUCCESSFULLY  🎉       ║${NC}"
echo -e "${MAGENTA}║                                                              ║${NC}"
echo -e "${MAGENTA}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${CYAN}Next Steps:${NC}"
echo ""
echo -e "${GREEN}1. Review Visual Results:${NC}"
echo -e "   open ${REPORTS_DIR}/model_comparison_dashboard.png"
echo ""
echo -e "${GREEN}2. Read Detailed Report:${NC}"
echo -e "   cat ${REPORTS_DIR}/improvement_report.md"
echo ""
echo -e "${GREEN}3. Use Optimized Model for Inference:${NC}"
echo -e "   python src/infer.py --config ${CONFIG_FILE} \\"
echo -e "       --image path/to/scan.jpg --threshold 0.65"
echo ""
echo -e "${GREEN}4. Deploy to Production:${NC}"
echo -e "   docker build -t brain-mri-focal:latest ."
echo -e "   docker run -p 8000:8000 brain-mri-focal:latest"
echo ""

echo -e "${YELLOW}📊 View all results in: ${REPORTS_DIR}/${NC}"
echo -e "${YELLOW}💾 Models saved in: ${CHECKPOINT_DIR}/${NC}"
echo ""
