# Changelog

Development history of Brain Tumor MRI Classification v2.

---

## v2.0.0 — Complete Implementation (March 2026)

### Phase 1 — Foundations (Weeks 1-2)

**Goal:** Repository setup, Hydra configuration, data loaders, critical leakage test.

**Added:**

- Project structure with 6 source subpackages following Section 5 design
- 13 Hydra YAML configs: root, 2 data, 4 model, 3 training, 3 experiment
- `src/data/splitter.py` — PatientLevelSplitter with StratifiedGroupKFold
- `src/data/transforms.py` — TorchIO MRI-specific augmentation pipeline
- `src/data/quality_control.py` — Automated dataset QC checks
- `src/data/datamodule.py` — Lightning DataModule with fold-aware setup
- `src/data/brats_dataset.py` — BraTS 2023 NIfTI dataset loader
- `src/data/ucsf_dataset.py` — UCSF-PDGM external validation loader
- `src/utils/reproducibility.py` — MONAI determinism + seed management
- `src/utils/logging_utils.py` — W&B experiment tracking helpers
- `scripts/download_data.py` — Synapse + TCIA download automation
- `tests/test_data/test_no_leakage.py` — 12 critical leakage tests
- `tests/test_data/test_splitter.py` — 10 splitter unit tests
- `tests/test_data/test_transforms.py` — 15 transform pipeline tests
- `dvc.yaml` — Complete reproducible DVC pipeline
- `Makefile` — 18 convenience commands
- `Dockerfile` — Container environment
- `pyproject.toml` — Pinned dependencies

**Tests:** 37 passing

---

### Phase 2 — Data Pipeline (Weeks 3-4)

**Goal:** Preprocessing scripts, label derivation, exploration notebooks.

**Added:**

- `src/data/preprocessing.py` — N4 bias correction, resampling, z-score, patient preprocessing
- `src/data/label_derivation.py` — BraTS/UCSF-PDGM grade label derivation
- `scripts/preprocess.py` — Preprocessing CLI with manifest generation
- `scripts/create_splits.py` — Split creation entry point
- `notebooks/01_data_exploration.py` — 7-section dataset exploration
- `notebooks/02_preprocessing_visualization.py` — Preprocessing effect visualization
- `tests/test_data/test_preprocessing.py` — 14 preprocessing tests
- `tests/test_data/test_label_derivation.py` — 10 label derivation tests

**Tests:** 61 passing (+24)

---

### Phase 3 — Models and Training (Weeks 5-6)

**Goal:** Backbone factory, classifier, training protocol, baselines.

**Added:**

- `src/models/backbones.py` — DenseNet-121, ResNet-50/18, EfficientNet-B0, SimpleCNN
- `src/models/heads.py` — Classification head with dropout
- `src/models/classifier.py` — Lightning module with two-phase training protocol
- `src/models/baseline_svm.py` — SVM + handcrafted features, DummyClassifier
- `src/training/losses.py` — FocalLoss, CE with label smoothing, class weights
- `src/training/callbacks.py` — GradualUnfreezeCallback, EarlyStopping, ModelCheckpoint
- `src/training/schedulers.py` — CosineAnnealingWithWarmup
- `scripts/train.py` — Single-fold training entry point
- `scripts/run_cross_validation.py` — 5-fold CV orchestrator
- `tests/test_models/test_forward_pass.py` — 14 backbone/classifier tests
- `tests/test_models/test_determinism.py` — 4 reproducibility tests

**Tests:** 79 passing (+18)

---

### Phase 4 — Rigorous Evaluation (Weeks 7-8)

**Goal:** CLAIM-compliant metrics, bootstrap CI, statistical tests.

**Added:**

- `src/evaluation/metrics.py` — ClassificationReport with all CLAIM metrics
- `src/evaluation/bootstrap.py` — Patient-level bootstrap CI (1000 iterations)
- `src/evaluation/statistical_tests.py` — McNemar, DeLong, Wilcoxon
- `src/evaluation/calibration.py` — ECE, Brier score, reliability diagrams
- `src/evaluation/report_generator.py` — LaTeX table generation, JSON reports
- `scripts/evaluate.py` — Evaluation CLI with all-folds and external modes
- `tests/test_evaluation/test_metrics.py` — 12 metric correctness tests
- `tests/test_evaluation/test_bootstrap.py` — 10 bootstrap CI tests
- `tests/test_evaluation/test_statistical_tests.py` — 12 statistical test tests

**Tests:** 112 passing (+34)

---

### Phase 5 — Experiments and Ablations (Weeks 9-10)

**Goal:** Split comparison, preprocessing ablation, naive experiment, figure generation.

**Added:**

- `src/experiments/split_comparison.py` — Image-level vs patient-level comparison
- `src/experiments/ablation_runner.py` — Generic ablation framework with LaTeX output
- `src/experiments/naive_experiment.py` — Kaggle approach reproduction and documentation
- `src/experiments/figure_generator.py` — 6 figure types at 300 DPI
- `scripts/compare_splits.py` — Split comparison CLI
- `scripts/ablation_preprocessing.py` — Preprocessing ablation CLI
- `scripts/run_naive_experiment.py` — Naive experiment CLI
- `scripts/generate_figures.py` — Master figure generation CLI
- `tests/test_experiments/test_split_comparison.py` — 8 leakage detection tests
- `tests/test_experiments/test_ablation_runner.py` — 8 ablation framework tests
- `tests/test_experiments/test_figure_generator.py` — 7 figure generation tests

**Tests:** 135 passing (+23)

---

### Phase 6 — Interpretability (Weeks 11-12)

**Goal:** Grad-CAM, IoU validation, XAI visualization.

**Added:**

- `src/interpretability/gradcam.py` — Captum Grad-CAM with manual fallback
- `src/interpretability/integrated_gradients.py` — Pixel-level attributions
- `src/interpretability/attention_validator.py` — IoU validation against ground truth
- `src/interpretability/visualization.py` — Publication-ready XAI figures
- `notebooks/03_kaggle_problems_analysis.py` — Kaggle dataset problem documentation
- `notebooks/04_interpretability_analysis.py` — IoU validation analysis
- `tests/test_interpretability/test_attention_validator.py` — 23 IoU validation tests
- `tests/test_interpretability/test_gradcam.py` — 10 Grad-CAM tests

**Tests:** 168 passing (+33)

---

### Phase 7 — Documentation (Weeks 13-14)

**Goal:** Comprehensive project documentation.

**Added:**

- `README.md` — Rewritten with quick start and documentation index
- `docs/ARCHITECTURE.md` — Technical architecture and design rationale
- `docs/DATA_GUIDE.md` — Data acquisition, preprocessing, validation
- `docs/EXPERIMENT_GUIDE.md` — Step-by-step experiment instructions
- `docs/METHODOLOGY.md` — Evidence-based methodological decisions
- `docs/REPRODUCIBILITY.md` — Environment and reproduction guide
- `docs/API_REFERENCE.md` — Module-by-module function reference
- `docs/CLAIM_CHECKLIST.md` — CLAIM/TRIPOD+AI compliance tracking
- `docs/CHANGELOG.md` — This file

---

## v1.0.0 — Initial Attempt (Deprecated)

The original v1 project used the Kaggle MasoudNickparvar dataset with image-level
splitting and achieved 99.2% accuracy. This was identified as artificial — the model
learned scanner artifacts and dataset source patterns rather than tumor morphology.

v1 was completely replaced by v2, which uses clinical-grade datasets (BraTS 2023,
UCSF-PDGM) with patient-level splitting and comprehensive evaluation following
CLAIM and TRIPOD+AI guidelines.

The discovery and correction of v1's methodological flaws became the central
narrative of the thesis: demonstrating that rigor matters more than numbers.
