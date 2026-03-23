# API Reference

Concise reference for every public function and class in the project.

---

## src.data

### splitter.PatientLevelSplitter

```python
splitter = PatientLevelSplitter(n_splits=5, shuffle=True, random_state=42)
splits = splitter.create_splits(patient_ids, labels, sample_indices=None)
splitter.save_splits("data/splits/brats_5fold.json")
loaded = PatientLevelSplitter.load_splits("data/splits/brats_5fold.json")
PatientLevelSplitter.verify_no_leakage(train_patients, val_patients, fold_name="fold_0")
```

### preprocessing

```python
from src.data.preprocessing import (
    apply_n4_bias_correction,    # (image_sitk, shrink_factor=4, iterations=[50,50,50,50]) → sitk.Image
    resample_volume,             # (image_sitk, target_spacing=(1,1,1), interpolation="bspline3") → sitk.Image
    z_score_normalize,           # (volume, foreground_only=True) → (normalized, mean, std)
    select_representative_slice, # (segmentation, volume_depth, method="max_tumor_area") → int
    preprocess_brats_patient,    # (patient_dir, output_dir, ...) → PreprocessingStats | None
    preprocess_ucsf_patient,     # (patient_dir, output_dir, ...) → PreprocessingStats | None
)
```

### label_derivation

```python
from src.data.label_derivation import (
    derive_brats_labels,               # (data_dir, metadata_path=None) → dict[str, int]
    derive_brats_labels_from_metadata, # (metadata_path) → dict[str, int]
    load_ucsf_labels,                  # (metadata_path) → dict[str, dict]
    save_labels,                       # (labels, output_path) → None
)
```

### transforms

```python
from src.data.transforms import (
    build_train_transforms,         # (cfg=None) → tio.Compose
    build_val_transforms,           # () → tio.Compose (empty/identity)
    build_preprocessing_transforms, # (z_score=True) → tio.Compose
    extract_2d_slice,               # (volume, segmentation, axis, method, context_slices) → (slice, idx)
)
```

### quality_control

```python
from src.data.quality_control import DatasetQualityControl, generate_qc_report

qc = DatasetQualityControl(expected_modalities=["t1n","t1c","t2w","t2f"], expected_shape=(240,240,155))
results = qc.check_dataset(data_dir)
report = generate_qc_report(results)
```

### datamodule.BrainTumorDataModule

```python
dm = BrainTumorDataModule(data_dir, splits_path, batch_size=32, fold=0)
dm.prepare_data()
dm.setup(stage="fit")
dm.set_fold(2)                      # Change active fold
dm.set_external_test(test_samples)  # Set UCSF-PDGM test set
```

---

## src.models

### backbones

```python
from src.models.backbones import build_backbone, count_parameters

backbone, feature_dim = build_backbone("densenet121", in_channels=4, pretrained=True)
# Options: "densenet121", "resnet50", "resnet18", "efficientnet_b0", "simple_cnn"

params = count_parameters(model)  # → {"total": int, "trainable": int}
```

### classifier.TumorClassifier

```python
model = TumorClassifier(
    backbone_name="densenet121", num_classes=2, in_channels=4,
    pretrained=True, head_hidden_dim=256, head_dropout=0.5,
    lr=1e-3, lr_backbone=1e-5, lr_head=1e-3, label_smoothing=0.1,
    freeze_backbone=True,
)
model.unfreeze_backbone()
groups = model.get_backbone_layer_groups()
```

### baseline_svm

```python
from src.models.baseline_svm import (
    extract_features_from_dataset,  # (samples) → (X, y, patient_ids)
    train_svm_baseline,             # (X_train, y_train) → Pipeline
    train_dummy_baseline,           # (X_train, y_train) → DummyClassifier
    evaluate_baseline,              # (model, X_test, y_test) → dict[str, float]
)
```

---

## src.training

### losses

```python
from src.training.losses import FocalLoss, build_loss, compute_class_weights

loss_fn = build_loss("focal_loss", gamma=2.0, class_weights=weights)
weights = compute_class_weights(labels)  # Inverse frequency weighting
```

### callbacks

```python
from src.training.callbacks import build_callbacks

callbacks = build_callbacks(
    monitor_metric="val/balanced_accuracy", patience=10,
    checkpoint_dir="outputs/models", enable_gradual_unfreeze=True,
)
```

### schedulers

```python
from src.training.schedulers import CosineAnnealingWithWarmup, build_scheduler

scheduler = build_scheduler(optimizer, name="cosine_annealing", warmup_epochs=3, max_epochs=35)
```

---

## src.evaluation

### metrics

```python
from src.evaluation.metrics import compute_full_report, compute_roc_curve_data

report = compute_full_report(y_true, y_pred, y_prob)
report.balanced_accuracy    # float
report.per_class_sensitivity  # dict[int, float]
report.confusion_matrix_absolute  # np.ndarray
report.to_dict()            # JSON-serializable dict
```

### bootstrap

```python
from src.evaluation.bootstrap import patient_level_bootstrap, bootstrap_single_metric

results = patient_level_bootstrap(y_true, y_pred, patient_ids, y_prob, n_iterations=1000)
results["balanced_accuracy"].format_str()  # "0.874 (0.831 — 0.912)"
```

### statistical_tests

```python
from src.evaluation.statistical_tests import mcnemar_test, delong_test, wilcoxon_signed_rank_test

result = mcnemar_test(y_true, y_pred_a, y_pred_b)   # TestResult
result = delong_test(y_true, y_prob_a, y_prob_b)     # TestResult
result = wilcoxon_signed_rank_test(scores_a, scores_b)  # TestResult
result.p_value, result.significant, result.effect_size
```

### calibration

```python
from src.evaluation.calibration import compute_calibration

cal = compute_calibration(y_true, y_prob, n_bins=10)
cal.ece, cal.brier_score, cal.bin_accuracies
```

---

## src.experiments

### split_comparison

```python
from src.experiments.split_comparison import run_split_comparison, detect_leakage_in_splits

result = run_split_comparison(labels, patient_ids, evaluate_fn, n_splits=5)
result.delta["balanced_accuracy"]  # Expected: negative (patient-level is harder)
```

### ablation_runner

```python
from src.experiments.ablation_runner import (
    run_ablation_study, define_preprocessing_ablations,
    define_architecture_ablations, define_slice_ablations,
)

study = run_ablation_study("Preprocessing", "Does each step help?",
                           baseline_config, variants, train_fn)
study.generate_latex_table()
```

### figure_generator

```python
from src.experiments.figure_generator import (
    plot_split_comparison, plot_ablation_study, plot_roc_curves,
    plot_confusion_matrix, plot_calibration_diagram, plot_cross_dataset_comparison,
)
```

---

## src.interpretability

### gradcam

```python
from src.interpretability.gradcam import GradCAMGenerator, binarize_heatmap, get_target_layer

layer = get_target_layer(model, "densenet121")
gen = GradCAMGenerator(model, layer, use_captum=True)
heatmap = gen.generate(image, target_class=1)  # np.ndarray (H, W) in [0, 1]
binary = binarize_heatmap(heatmap, threshold_ratio=0.5)
```

### attention_validator

```python
from src.interpretability.attention_validator import (
    validate_attention, validate_single_sample, compute_iou, compute_dice,
)

summary = validate_attention(model, dataloader, gradcam_gen, correct_only=True)
summary.mean_iou        # float
summary.interpretation   # "GOOD: ..." / "MIXED: ..." / "WARNING: ..."
```
