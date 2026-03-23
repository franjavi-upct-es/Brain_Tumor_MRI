# Methodology

Every methodological decision in this project is evidence-based. This document
maps each decision to the literature that supports it.

---

## 1. Patient-Level Splitting

**Decision:** All data splits use `StratifiedGroupKFold` by patient ID.

**Evidence:**
- Yagis et al. (2021, *Scientific Reports*): slice-level CV inflated accuracy by 30% on OASIS and 48% on PPMI. On randomly labeled data, slice-level splitting still yielded ~96% accuracy — proving the model learned patient identity, not disease.
- Wen et al. (2020, *Medical Image Analysis*): >50% of Alzheimer's classification papers had data leakage, with accuracy dropping from 100% to 79% when corrected.
- A 2025 Springer paper explicitly identified patient-wise splitting as "a major limitation in existing tumor-classification studies."

**Implementation:** `src/data/splitter.py`, verified by 12 tests in `tests/test_data/test_no_leakage.py`.

---

## 2. Z-Score Normalization (Foreground Only)

**Decision:** Per-volume z-score normalization computed on voxels > 0.

**Evidence:**
- Isensee et al. (2021, *Nature Methods*): this is the nnU-Net default for MRI data.
- Carré et al. (2020, *Scientific Reports*): tested three normalization methods on tumor grade classification (n=130); all significantly improved accuracy over no normalization. Z-score performed equivalently to WhiteStripe and Nyúl.
- Federated learning study found z-score normalized datasets were "most compatible" across differently-preprocessed models.

**Implementation:** `src/data/preprocessing.py::z_score_normalize()`.

---

## 3. N4ITK Bias Field Correction

**Decision:** Apply N4ITK (Tustison et al., 2010) on UCSF-PDGM data with parameters shrinkFactor=4, iterations=[50,50,50,50].

**Evidence:**
- Tustison et al. (2010, *IEEE TMI*): N4ITK is the gold standard for MRI bias correction, improving upon N3.
- Foltyn-Dumitru et al. (2023, *European Radiology*): N4 + intensity normalization improved glioma molecular subtype prediction AUC from 0.84 to 0.87 on 615 patients, validated externally.

**Implementation:** `src/data/preprocessing.py::apply_n4_bias_correction()`.

---

## 4. MRI-Specific Augmentation (TorchIO)

**Decision:** Use TorchIO for physically motivated augmentations. Do NOT use color jitter, vertical flip, or CutMix/MixUp.

**Evidence:**
- Pérez-García et al. (2021, *CMPB*): TorchIO provides augmentations that simulate real MRI acquisition artifacts (B0 field inhomogeneity, motion ghosting, thermal noise).
- Brain anatomy is sagittally symmetric but NOT vertically symmetric — vertical flip creates anatomically impossible images.
- CutMix/MixUp are not validated in medical imaging and can create pathologically unrealistic combinations.
- Conservative parameter ranges (±15° rotation, ±10% scaling) validated in BraTS challenge submissions.

**Implementation:** `src/data/transforms.py::build_train_transforms()`.

---

## 5. Two-Phase Training Protocol

**Decision:** Phase 1 (head-only, backbone frozen, 10 epochs) → Phase 2 (gradual unfreezing with discriminative LR, 25 epochs).

**Evidence:**
- Howard & Ruder (2018, ULMFiT): gradual unfreezing prevents catastrophic forgetting of pretrained features.
- Raghu et al. (2019, *NeurIPS*): the main benefit of ImageNet pretraining on medical images is convergence speed, not final accuracy. This motivates a short Phase 1 for fast convergence followed by Phase 2 for task-specific refinement.
- Discriminative LR (low for backbone, high for head) preserves general features while allowing the head to specialize.

**Implementation:** `src/models/classifier.py`, `src/training/callbacks.py::GradualUnfreezeCallback`.

---

## 6. Balanced Accuracy as Primary Metric

**Decision:** Report balanced accuracy (macro-averaged recall) as the primary metric instead of standard accuracy.

**Evidence:**
- CLAIM checklist (Tejani et al., 2024, *Radiology: AI*): requires reporting metrics that correct for class imbalance.
- UCSF-PDGM has severe imbalance (403 Grade IV vs 55 Grade II). Standard accuracy on this dataset would be 81.4% even for a majority-class predictor.
- Balanced accuracy treats each class equally regardless of size.

**Implementation:** Primary metric in `src/evaluation/metrics.py` and all evaluation scripts.

---

## 7. Patient-Level Bootstrap CI

**Decision:** All confidence intervals computed via patient-level bootstrap (1000 iterations, 95% CI).

**Evidence:**
- Efron & Tibshirani (1993): bootstrap provides reliable CIs without distributional assumptions.
- Multiple slices from one patient are correlated. Slice-level resampling underestimates CI width. Patient-level resampling correctly accounts for within-patient correlation.
- TRIPOD+AI (Collins et al., 2024, *BMJ*): recommends reporting confidence intervals for all performance metrics.

**Implementation:** `src/evaluation/bootstrap.py::patient_level_bootstrap()`.

---

## 8. Grad-CAM IoU Validation

**Decision:** Validate Grad-CAM heatmaps against ground truth tumor segmentations using IoU, with interpretation thresholds at 0.3 and 0.5.

**Evidence:**
- DeGrave et al. (2021, *Nature Machine Intelligence*): COVID-19 detection models learned laterality markers and text annotations, not lung pathology. This was only discovered through attribution analysis.
- Zech et al. (2018, *PLOS Medicine*): CNNs achieved >99% accuracy at identifying hospital systems from chest X-rays — learning scanner signatures, not disease.
- IoU provides a quantitative, reproducible measure of attention overlap that goes beyond qualitative visual inspection.

**Implementation:** `src/interpretability/attention_validator.py`.

---

## 9. Architecture Selection

**Decision:** DenseNet-121 as primary model, with ResNet-50, EfficientNet-B0, ResNet-18, and SimpleCNN as comparisons.

**Evidence:**
- DenseNet-121: MONAI's preferred classification architecture. Used in RadImageNet (Mei et al., 2022, *Radiology: AI*). ~8M parameters, efficient feature reuse.
- ResNet-50: most widely used in the literature. Facilitates direct comparison with published results.
- EfficientNet-B0: best computational efficiency-to-performance ratio in several comparisons. ~5M parameters.
- SimpleCNN (4-layer from scratch): Raghu et al. (2019) found simple CNNs performed comparably to large pretrained models. Including this baseline measures the actual contribution of transfer learning.
- ResNet-18: standard lightweight pretrained baseline.

**Implementation:** `src/models/backbones.py`.

---

## 10. Mandatory Baselines

**Decision:** Include DummyClassifier (majority class) and SVM + handcrafted features as non-DL baselines.

**Evidence:**
- A model that cannot significantly outperform a majority-class predictor has learned nothing.
- SVM with radiomics features represents what a computational radiologist would use without deep learning. This establishes the DL value-add.
- These baselines are required by the CLAIM checklist for complete reporting.

**Implementation:** `src/models/baseline_svm.py`.

---

## 11. Statistical Significance Testing

**Decision:** McNemar's test for error rate comparison, DeLong's test for AUC comparison, Wilcoxon signed-rank for fold-level comparison.

**Evidence:**
- McNemar (1947): the standard test for comparing two classifiers on the same test set. Tests whether discordant errors are symmetric.
- DeLong et al. (1988): nonparametric comparison of correlated ROC curves. Standard in medical imaging evaluation.
- Wilcoxon signed-rank: distribution-free test for paired observations. Appropriate for comparing metrics across CV folds (5 paired observations).

**Implementation:** `src/evaluation/statistical_tests.py`.
