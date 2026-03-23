# Experiment Guide

Step-by-step instructions for running every experiment in the project.

---

## Prerequisites

All experiments require preprocessed data and splits:
```bash
make preprocess   # ~1-2 hours depending on dataset size
make splits       # ~5 seconds
```

---

## 1. Single Fold Training (Smoke Test)

Train one fold to verify the pipeline works before committing to full CV:

```bash
python scripts/train.py --fold 0 --backbone densenet121 --phase1-epochs 5 --phase2-epochs 10
```

Expected output: `outputs/models/fold_0/` and `outputs/metrics/fold_0/metrics.json`

---

## 2. Full 5-Fold Cross-Validation

```bash
python scripts/run_cross_validation.py --backbone densenet121
# Or train each fold manually:
make train
```

This runs the two-phase protocol for all 5 folds and produces:
- Per-fold models in `outputs/models/fold_{0-4}/`
- Per-fold metrics in `outputs/metrics/fold_{0-4}/metrics.json`
- Aggregated summary in `outputs/evaluation/cv_summary.json`

**Training multiple architectures:**
```bash
python scripts/run_cross_validation.py --backbone densenet121
python scripts/run_cross_validation.py --backbone resnet50
python scripts/run_cross_validation.py --backbone efficientnet_b0
python scripts/run_cross_validation.py --backbone simple_cnn
```

---

## 3. Image-Level vs Patient-Level Split Comparison

**THE key ablation demonstrating data leakage impact.**

```bash
python scripts/compare_splits.py
```

Output:
- `outputs/evaluation/split_comparison.json` — metrics for both strategies
- `outputs/evaluation/leakage_analysis.json` — quantified leakage
- `outputs/figures/split_comparison.png` — bar chart visualization

Expected finding: image-level accuracy is 7–30 pp higher than patient-level.

---

## 4. Preprocessing Ablation Study

Removes one preprocessing step at a time to measure contribution:

```bash
python scripts/ablation_preprocessing.py --backbone densenet121
```

Output:
- `outputs/evaluation/ablation_preprocessing.json`
- `outputs/evaluation/ablation_preprocessing.tex` — LaTeX table
- `outputs/figures/ablation_preprocessing.png`

Variants tested: no N4, no z-score, no augmentation, generic augmentation, no pretraining.

---

## 5. Naive Experiment (Chapter 3 Reproduction)

Reproduces the flawed Kaggle methodology for comparison:

```bash
python scripts/run_naive_experiment.py --kaggle-dir data/raw/kaggle
```

This documents the problems and generates comparison data. The Kaggle dataset is optional — the script generates documented evidence of the issues even without the data.

---

## 6. External Validation on UCSF-PDGM

Test generalization on a completely independent dataset:

```bash
python scripts/evaluate.py --dataset ucsf_pdgm --bootstrap 1000
```

Expected: 3-8 percentage point drop from BraTS performance (this is normal and demonstrates honest generalization measurement).

---

## 7. Full Evaluation with Bootstrap CI

```bash
python scripts/evaluate.py --all-folds --bootstrap 1000
```

Computes for each fold:
- Balanced accuracy, macro F1, AUC-ROC, Cohen's kappa (with 95% CI)
- ECE, Brier score, reliability diagram data
- ROC curve data for plotting
- Confusion matrices (absolute and normalized)

---

## 8. Generate All Thesis Figures

```bash
python scripts/generate_figures.py --all
```

Generates every figure at 300 DPI in `outputs/figures/`:
- `ch4_split_comparison.png` — image-level vs patient-level
- `ch6_ablation_preprocessing.png` — ablation bar chart
- `ch6_roc_curves.png` — multi-model ROC overlay
- `ch6_cm_*.png` — confusion matrix heatmaps
- `ch6_calibration_*.png` — reliability diagrams
- `ch6_cross_dataset.png` — BraTS vs UCSF-PDGM comparison

---

## 9. Interpretability Analysis

```bash
# Run IoU validation demonstration (works without trained models)
python notebooks/04_interpretability_analysis.py

# Run full Grad-CAM IoU validation (requires trained models)
python scripts/generate_figures.py --gradcam --iou-validation
```

---

## 10. Exploration Notebooks

```bash
python notebooks/01_data_exploration.py --data-dir data/raw/brats2023
python notebooks/02_preprocessing_visualization.py
python notebooks/03_kaggle_problems_analysis.py
python notebooks/04_interpretability_analysis.py
```

---

## DVC Pipeline (Reproducible End-to-End)

Run the entire pipeline with tracked dependencies:

```bash
dvc repro           # Runs everything in dependency order
dvc metrics show    # Display all tracked metrics
dvc plots show      # Generate comparison plots
```

The `dvc.yaml` defines stages: download → preprocess → split → train (5 folds) → evaluate → external validation → interpretability.

---

## W&B Experiment Tracking

Enable Weights & Biases logging:

```bash
wandb login
python scripts/train.py --fold 0 --wandb-mode online
```

All metrics, learning curves, and hyperparameters are logged automatically.
