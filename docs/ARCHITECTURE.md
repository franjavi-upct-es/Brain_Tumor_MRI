# Technical Architecture

This document describes every module in the project, explains the design
decisions behind each, and maps the data flow from raw NIfTI volumes to
publication-ready evaluation reports.

---

## High-Level Data Flow

```
Raw NIfTI (.nii.gz)
    │
    ├─ BraTS 2023: already co-registered, 1mm³, skull-stripped
    │  └─ preprocessing.py: QC → z-score → slice selection → NPZ
    │
    └─ UCSF-PDGM: requires full preprocessing
       └─ preprocessing.py: N4 bias correction → resample → z-score → slice → NPZ
    │
    ▼
Preprocessed NPZ files + manifest.json
    │
    ├─ create_splits.py: StratifiedGroupKFold → brats_5fold.json
    │
    ▼
DataModule (datamodule.py)
    │
    ├─ Training split: SliceDataset + TorchIO augmentation
    └─ Validation split: SliceDataset + identity transform (NO augmentation)
    │
    ▼
TumorClassifier (classifier.py)
    │
    ├─ Backbone: DenseNet-121 / ResNet-50 / EfficientNet-B0 / SimpleCNN
    └─ Head: Linear(features → hidden → num_classes)
    │
    ▼
Training (train.py)
    │
    ├─ Phase 1: head-only, backbone frozen, CE + label smoothing
    └─ Phase 2: gradual unfreezing, discriminative LR, cosine schedule
    │
    ▼
Evaluation (evaluate.py)
    │
    ├─ metrics.py: ClassificationReport (balanced acc, F1, AUC, kappa, confusion matrix)
    ├─ bootstrap.py: patient-level 95% CI (1000 iterations)
    ├─ calibration.py: ECE, Brier score, reliability diagram
    └─ statistical_tests.py: McNemar, DeLong, Wilcoxon
    │
    ▼
Interpretability (gradcam.py → attention_validator.py)
    │
    ├─ Generate Grad-CAM heatmaps via Captum
    ├─ Binarize at 0.5 × max
    ├─ Compare vs ground truth segmentation → IoU
    └─ Aggregate: mean IoU per class with interpretation thresholds
    │
    ▼
Figures and Tables (figure_generator.py, report_generator.py)
    │
    ├─ PNG figures at 300 DPI
    └─ LaTeX tables with bootstrap CI
```

---

## Module Map

### `src/data/` — Data Pipeline

| Module | Purpose |
|--------|---------|
| `brats_dataset.py` | PyTorch Dataset for BraTS 2023. Discovers patient dirs, loads 4 modalities + segmentation as NIfTI, applies z-score normalization, extracts representative 2D slice. |
| `ucsf_dataset.py` | PyTorch Dataset for UCSF-PDGM external validation. Mirrors BraTSDataset interface for cross-dataset evaluation. Loads clinical metadata (grade, IDH, MGMT). |
| `preprocessing.py` | Offline preprocessing engine. `apply_n4_bias_correction()` via SimpleITK, `resample_volume()` to 1mm³, `z_score_normalize()` on foreground, `preprocess_brats_patient()` and `preprocess_ucsf_patient()` end-to-end. |
| `label_derivation.py` | Grade label derivation. Reads BraTS metadata CSV or falls back to segmentation heuristic (enhancing tumor voxel count). UCSF-PDGM labels from clinical CSV. |
| `splitter.py` | Patient-level StratifiedGroupKFold wrapper. Guarantees zero patient overlap between folds (assertion-verified). Saves/loads splits as JSON. |
| `transforms.py` | TorchIO augmentation pipelines. Training: affine, sagittal flip, bias field, noise, ghosting, gamma. Validation: identity (NO augmentation). Also handles 2D/2.5D slice extraction. |
| `quality_control.py` | Automated QC: checks dimensions, modality completeness, empty volumes, minimum tumor voxels. Generates JSON reports. |
| `datamodule.py` | Lightning DataModule. Fold-aware setup, integrates splits + transforms + datasets. Verifies no leakage at runtime. |

### `src/models/` — Classification Models

| Module | Purpose |
|--------|---------|
| `backbones.py` | Factory for feature extractors. DenseNet-121 (MONAI), ResNet-50/18 (torchvision), EfficientNet-B0 (timm), SimpleCNN (from scratch). Adapts first conv layer for 4-channel MRI input. |
| `heads.py` | Classification head: Linear → ReLU → Dropout → Linear. Configurable hidden dim and dropout rate. |
| `classifier.py` | Lightning module. Two-phase training protocol, discriminative LR, CLAIM metric tracking (balanced accuracy, F1, AUC, kappa), backbone freeze/unfreeze with layer groups. |
| `baseline_svm.py` | Non-DL baselines. Handcrafted feature extraction (intensity stats, gradient texture, shape). SVM with RBF kernel. DummyClassifier (majority class floor). |

### `src/training/` — Training Infrastructure

| Module | Purpose |
|--------|---------|
| `losses.py` | Cross-entropy with label smoothing. FocalLoss (gamma=2.0, per-class alpha). Inverse-frequency class weight computation. |
| `callbacks.py` | EarlyStopping (patience=10 on balanced accuracy), ModelCheckpoint (best AUC), LRMonitor, GradualUnfreezeCallback (progressive layer unfreezing). |
| `schedulers.py` | CosineAnnealingWithWarmup (3-epoch linear warmup, cosine decay to 1e-7). |

### `src/evaluation/` — Evaluation Framework

| Module | Purpose |
|--------|---------|
| `metrics.py` | `compute_full_report()` → ClassificationReport with all CLAIM metrics. Per-class sensitivity/specificity/precision/F1/AUC. Confusion matrices (absolute + normalized). ROC curve data. |
| `bootstrap.py` | Patient-level bootstrap CI (1000 iterations, 95% CI). Resamples patients, not slices. `BootstrapResult` with format_str producing "0.874 (0.831 — 0.912)". |
| `statistical_tests.py` | McNemar (error rate comparison), DeLong (AUC comparison with nonparametric variance), Wilcoxon signed-rank (paired fold comparison). All return TestResult with p-value and effect size. |
| `calibration.py` | ECE, Brier score, reliability diagram bin data. Handles binary and multi-class. |
| `report_generator.py` | LaTeX table generation (main results, confusion matrix, naive vs rigorous comparison). JSON report export. `format_metric_with_ci()` for text output. |

### `src/experiments/` — Experiment Orchestration

| Module | Purpose |
|--------|---------|
| `split_comparison.py` | Image-level vs patient-level split comparison. Creates both split types, detects and quantifies leakage, runs evaluation on both, computes deltas. |
| `ablation_runner.py` | Generic ablation framework. Runs baseline + N variants, computes deltas, runs Wilcoxon tests, generates LaTeX tables. Predefined ablation definitions for preprocessing, architecture, and slice strategy. |
| `naive_experiment.py` | Reproduces the flawed Kaggle approach. Documents 6 dataset problems with citations. Generates naive vs rigorous comparison data. |
| `figure_generator.py` | 6 figure types at 300 DPI: split comparison, ablation bars, ROC curves, confusion matrix heatmap, calibration diagram, cross-dataset comparison. |

### `src/interpretability/` — Explainability

| Module | Purpose |
|--------|---------|
| `gradcam.py` | Grad-CAM generator (Captum backend + manual fallback). Binarization at 0.5×max. Automatic target layer detection for all backbone architectures. |
| `integrated_gradients.py` | Pixel-resolution attributions via Captum IntegratedGradients (50 steps from zero baseline). Complementary to Grad-CAM. |
| `attention_validator.py` | THE key experiment: IoU between binarized Grad-CAM and ground truth tumor segmentation. Three-tier interpretation (>0.5 good, 0.3-0.5 mixed, <0.3 shortcut). Per-class aggregation. |
| `visualization.py` | Publication figures: Grad-CAM overlay with segmentation contour, naive-vs-rigorous side-by-side, IoU distribution histogram, multi-patient validation grid. |

---

## Design Decisions

### Why patient-level splitting is the default

Every split in this project uses `StratifiedGroupKFold` with patient IDs as groups. This is enforced by:
1. The `PatientLevelSplitter` class which asserts zero overlap.
2. The `DataModule.setup()` which calls `verify_no_leakage()` at runtime.
3. 12 dedicated tests in `test_no_leakage.py`.
4. The `compare_splits.py` experiment that quantifies the inflation from image-level splitting.

### Why z-score normalization uses foreground only

MRI background (air) is zero-valued. Including it in mean/std computation skews normalization. The nnU-Net standard (Isensee et al., 2021) computes statistics on voxels > 0 only. This is the default for both BraTS and UCSF-PDGM preprocessing.

### Why augmentation uses TorchIO instead of albumentations

TorchIO provides physically motivated MRI augmentations (bias field, ghosting, motion artifacts) that simulate real acquisition noise. Generic augmentations (color jitter, hue shift) have no physical meaning in MRI. The ablation study quantifies this difference.

### Why the first conv layer is adapted instead of using 3 channels

MRI provides 4 modalities (T1, T1ce, T2, FLAIR) that each contribute different diagnostic information. Dropping channels or converting to RGB loses clinical signal. The adaptation strategy averages pretrained RGB weights and replicates to 4 channels, preserving learned low-level features.

### Why bootstrap CIs are patient-level

Multiple slices from the same patient are correlated. Resampling at the slice level underestimates confidence interval width because it treats correlated observations as independent. Patient-level bootstrap correctly accounts for this clustering.
