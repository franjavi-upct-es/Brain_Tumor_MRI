"""Focused coverage tests for lightly exercised modules."""

from __future__ import annotations

import json
import sys
import types
import builtins
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from torch import nn


def _write_nifti(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), str(path))


def test_brats_dataset_loads_and_exposes_metadata(tmp_path: Path) -> None:
    from src.data.brats_dataset import BraTSDataset, MODALITY_SUFFIXES, SEG_SUFFIX

    patient_id = "BraTS-GLI-00001"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()
    volume = np.zeros((4, 5, 3), dtype=np.float32)
    volume[:, :, 0] = 1
    volume[:, :, 1] = np.arange(20, dtype=np.float32).reshape(4, 5) + 1
    for suffix in MODALITY_SUFFIXES.values():
        _write_nifti(patient_dir / f"{patient_id}{suffix}", volume)
    seg = np.zeros((4, 5, 3), dtype=np.float32)
    seg[1:3, 1:4, 1] = 1
    _write_nifti(patient_dir / f"{patient_id}{SEG_SUFFIX}", seg)

    ignored_dir = tmp_path / "unlabeled"
    ignored_dir.mkdir()
    dataset = BraTSDataset(str(tmp_path), {patient_id: 1}, context_slices=1)

    sample = dataset[0]

    assert len(dataset) == 1
    assert dataset.get_patient_ids() == [patient_id]
    np.testing.assert_array_equal(dataset.get_labels(), np.array([1]))
    assert sample["image"].shape == (12, 4, 5)
    assert sample["label"] == 1
    assert sample["slice_idx"] == 1
    assert sample["segmentation"].sum() == 6

    dataset.transform = lambda sample: {**sample, "transformed": True}
    assert dataset[0]["transformed"] is True
    dataset.slice_method = "center"
    assert dataset._select_slice(np.zeros((2, 2, 4), dtype=np.int32)) == 2
    dataset.slice_method = "max_tumor_area"
    assert dataset._select_slice(np.zeros((2, 2, 4), dtype=np.int32)) == 2
    dataset.slice_method = "bad"
    with pytest.raises(ValueError):
        dataset._select_slice(np.zeros((2, 2, 4), dtype=np.int32))
    np.testing.assert_array_equal(
        dataset._apply_z_score(np.zeros((2, 2, 2), dtype=np.float32)),
        np.zeros((2, 2, 2), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        dataset._apply_z_score(np.ones((2, 2, 2), dtype=np.float32)),
        np.ones((2, 2, 2), dtype=np.float32),
    )


def test_brats_dataset_validation_paths(tmp_path: Path) -> None:
    from src.data.brats_dataset import BraTSDataset, MODALITY_SUFFIXES, SEG_SUFFIX

    with pytest.raises(FileNotFoundError):
        BraTSDataset(str(tmp_path / "missing"), {})

    patient_id = "p001"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()
    for suffix in list(MODALITY_SUFFIXES.values())[:-1]:
        (patient_dir / f"{patient_id}{suffix}").touch()
    (patient_dir / f"{patient_id}{SEG_SUFFIX}").touch()
    with pytest.raises(RuntimeError, match="No valid patients"):
        BraTSDataset(str(tmp_path), {patient_id: 0})


def test_quality_control_reports_pass_and_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.data.quality_control import DatasetQualityControl, generate_qc_report

    patient_dir = tmp_path / "patient_1"
    patient_dir.mkdir()
    for modality in ["t1n", "t1c", "t2w", "t2f"]:
        data = np.ones((3, 3, 3), dtype=np.float32)
        _write_nifti(patient_dir / f"patient_1_{modality}.nii.gz", data)
    seg = np.zeros((3, 3, 3), dtype=np.float32)
    seg[0, 0, 0] = 1
    _write_nifti(patient_dir / "patient_1_seg.nii.gz", seg)

    qc = DatasetQualityControl(expected_shape=(3, 3, 3), min_tumor_voxels=1)
    result = qc.check_patient(patient_dir)
    assert result.passed is True
    assert result.modalities_found == ["t1n", "t1c", "t2w", "t2f"]
    assert result.volume_shape == (3, 3, 3)
    assert result.tumor_voxels == 1

    failing = tmp_path / "patient_2"
    failing.mkdir()
    _write_nifti(failing / "patient_2_t1n.nii.gz", np.zeros((2, 2, 2)))
    failed = DatasetQualityControl(
        expected_shape=(3, 3, 3), min_tumor_voxels=2
    ).check_patient(failing)
    assert failed.passed is False
    assert any("Missing modality" in issue for issue in failed.issues)
    assert any("Unexpected shape" in issue for issue in failed.issues)
    assert any("Empty volume" in issue for issue in failed.issues)

    dataset_results = qc.check_dataset(tmp_path)
    report = generate_qc_report(dataset_results)
    assert report["total_patients"] == 2
    assert report["passed"] == 1
    assert report["failed"] == 1

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    assert qc.check_dataset(empty_root) == []

    broken = tmp_path / "broken"
    broken.mkdir()
    for name in ["broken_t1n.nii.gz", "broken_seg.nii.gz"]:
        (broken / name).touch()
    original_load = nib.load

    def broken_load(path):
        if "broken" in str(path):
            raise RuntimeError("cannot read")
        return original_load(path)

    monkeypatch.setattr(nib, "load", broken_load)
    broken_result = DatasetQualityControl(expected_shape=(3, 3, 3)).check_patient(
        broken
    )
    assert any("Cannot load" in issue for issue in broken_result.issues)
    assert any("Error reading" in issue for issue in broken_result.issues)


def test_datamodule_manifest_directory_and_dataloaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.data.datamodule import BrainTumorDataModule, SliceDataset

    image = np.ones((1, 3, 3), dtype=np.float32)
    np.savez(tmp_path / "sample_a.npz", image=image, label=0, patient_id="a")
    np.savez(tmp_path / "sample_b.npz", image=image * 2, label=1)

    manifest_dir = tmp_path / "manifested"
    manifest_dir.mkdir()
    np.savez(manifest_dir / "a.npz", image=image, label=0)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "file": "manifested/a.npz",
                        "label": 0,
                        "patient_id": "a",
                        "slice_index": 2,
                    }
                ]
            }
        )
    )

    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        json.dumps(
            {
                "folds": [
                    {"train_indices": [0], "val_indices": [1]},
                    {"train_indices": [1], "val_indices": [0]},
                ]
            }
        )
    )
    monkeypatch.setattr("src.data.datamodule.build_train_transforms", lambda cfg: None)
    monkeypatch.setattr("src.data.datamodule.build_val_transforms", lambda: None)

    dm = BrainTumorDataModule(
        tmp_path, splits_path, batch_size=1, num_workers=0, pin_memory=False
    )
    dm.prepare_data()
    directory_samples = dm._load_from_directory()
    assert [sample["patient_id"] for sample in directory_samples] == ["a", "sample_b"]
    assert dm._load_from_manifest(manifest_dir / "manifest.json")[0]["slice_index"] == 2

    dm.setup("fit")
    assert isinstance(dm.train_dataset, SliceDataset)
    assert next(iter(dm.train_dataloader()))["image"].shape == (1, 1, 3, 3)
    assert next(iter(dm.val_dataloader()))["label"].item() == 1

    dm.set_external_test([directory_samples[0]])
    assert next(iter(dm.test_dataloader()))["patient_id"][0] == "a"
    dm.set_fold(1)
    assert dm.train_dataset is None
    with pytest.raises(ValueError):
        dm.set_fold(99)

    test_only = BrainTumorDataModule(
        tmp_path, splits_path, batch_size=1, num_workers=0, pin_memory=False
    )
    test_only.setup("test")
    assert test_only.val_dataset is not None

    with pytest.raises(FileNotFoundError):
        BrainTumorDataModule(tmp_path / "missing", splits_path).prepare_data()
    with pytest.raises(FileNotFoundError):
        BrainTumorDataModule(tmp_path, tmp_path / "missing.json").prepare_data()


def test_preprocessing_brats_success_and_failures(tmp_path: Path) -> None:
    from src.data.brats_dataset import MODALITY_SUFFIXES, SEG_SUFFIX
    from src.data.preprocessing import preprocess_brats_patient

    patient_id = "patient"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()
    volume = np.ones((3, 3, 3), dtype=np.float32)
    volume[:, :, 1] = 2
    for suffix in MODALITY_SUFFIXES.values():
        _write_nifti(patient_dir / f"{patient_id}{suffix}", volume)
    seg = np.zeros((3, 3, 3), dtype=np.float32)
    seg[:, :, 0] = 1
    _write_nifti(patient_dir / f"{patient_id}{SEG_SUFFIX}", seg)

    stats = preprocess_brats_patient(
        patient_dir,
        tmp_path / "out",
        expected_shape=(3, 3, 3),
        context_slices=1,
    )
    assert stats is not None
    assert stats.selected_slice == 0
    assert stats.final_shape == (12, 3, 3)
    assert (tmp_path / "out" / "patient.npz").exists()

    assert (
        preprocess_brats_patient(
            patient_dir, tmp_path / "out2", modalities=["t1"], expected_shape=(2, 2, 2)
        )
        is None
    )


def test_label_derivation_mapping_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.data import label_derivation as labels_module
    from src.data.brats_dataset import SEG_SUFFIX

    pd = pytest.importorskip("pandas")
    mapping = tmp_path / "BraTS2023_2017_GLI_Mapping.xlsx"
    mapping.touch()
    monkeypatch.setattr(
        pd,
        "read_excel",
        lambda path: pd.DataFrame(
            {
                "BraTS2023": ["BraTS-known", "BraTS-low", "BraTS-skip"],
                "Cohort Name (if publicly available)": [
                    "TCGA-GBM",
                    "TCGA-LGG",
                    "Private Collection",
                ],
            }
        ),
    )
    mapped = labels_module.derive_brats_labels_from_mapping_xlsx(mapping)
    assert mapped == {"BraTS-known": 1, "BraTS-low": 0}

    bad_mapping = labels_module.derive_brats_labels_from_mapping_xlsx(mapping)
    assert bad_mapping == mapped

    patient_dir = tmp_path / "BraTS-missing"
    patient_dir.mkdir()
    seg = np.zeros((2, 2, 2), dtype=np.float32)
    seg[:, :, 0] = 3
    _write_nifti(patient_dir / f"BraTS-missing{SEG_SUFFIX}", seg)

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("ID,Grade\nBraTS-csv,LGG\n")
    combined = labels_module.derive_brats_labels(tmp_path, metadata_path=csv_path)
    assert combined["BraTS-known"] == 1
    assert combined["BraTS-low"] == 0
    assert combined["BraTS-csv"] == 0
    assert combined["BraTS-missing"] == 0

    monkeypatch.setattr(
        pd,
        "read_excel",
        lambda path: pd.DataFrame({"wrong": ["x"]}),
    )
    assert labels_module.derive_brats_labels_from_mapping_xlsx(mapping) == {}


def test_naive_experiment_helpers(tmp_path: Path) -> None:
    from src.experiments.naive_experiment import (
        NaiveExperimentResult,
        analyze_kaggle_dataset,
        create_naive_splits,
        generate_naive_comparison_data,
    )

    train, val, test = create_naive_splits(10, seed=1)
    assert len(train) == 8
    assert len(val) == 1
    assert len(test) == 1

    missing = analyze_kaggle_dataset(tmp_path / "missing")
    assert missing["status"] == "data_not_available"

    (tmp_path / "glioma").mkdir()
    (tmp_path / "glioma" / "a.jpg").touch()
    (tmp_path / "glioma" / "b.jpeg").touch()
    analysis = analyze_kaggle_dataset(tmp_path)
    assert analysis["class_counts"]["glioma"] == 2
    assert analysis["total_images"] == 2

    result = NaiveExperimentResult(
        accuracy=0.99,
        val_accuracy=0.98,
        grad_cam_iou=0.1,
        external_accuracy=0.6,
    )
    comparison = generate_naive_comparison_data(result, {"balanced_accuracy": 0.87})
    assert result.to_dict()["accuracy"] == 0.99
    assert comparison["naive"]["grad_cam_iou"] == 0.1
    assert comparison["rigorous"]["balanced_accuracy"] == 0.87


def test_baseline_svm_feature_training_and_auc_fallback() -> None:
    from src.models.baseline_svm import (
        evaluate_baseline,
        extract_features_from_dataset,
        extract_handcrafted_features,
        train_dummy_baseline,
        train_svm_baseline,
    )

    image = np.stack(
        [np.arange(1, 17).reshape(4, 4), np.arange(17, 33).reshape(4, 4)]
    ).astype(np.float32)
    features = extract_handcrafted_features(image)
    assert features.shape == (26,)
    assert np.isfinite(features).all()

    samples = [
        {"image": torch.tensor(image), "label": 0, "patient_id": "a"},
        {"image": image + 1, "label": 1, "patient_id": "b"},
        {"image": image + 2, "label": 1, "patient_id": "c"},
    ]
    x, y, patient_ids = extract_features_from_dataset(samples)
    assert x.shape == (3, 26)
    assert patient_ids == ["a", "b", "c"]

    model = train_dummy_baseline(x, y)
    metrics = evaluate_baseline(model, x, y, "dummy")
    assert "balanced_accuracy" in metrics
    assert "f1_macro" in metrics

    svm = train_svm_baseline(
        np.array([[0.0], [1.0], [2.0], [3.0]]),
        np.array([0, 0, 1, 1]),
        kernel="linear",
    )
    svm_metrics = evaluate_baseline(
        svm, np.array([[0.0], [3.0]]), np.array([0, 1]), "svm"
    )
    assert svm_metrics["balanced_accuracy"] == 1.0

    class BadProbaModel:
        def predict(self, x):
            return np.zeros(len(x), dtype=int)

        def predict_proba(self, x):
            raise RuntimeError("no probabilities")

    fallback_metrics = evaluate_baseline(
        BadProbaModel(), np.zeros((2, 1)), np.array([0, 1]), "bad"
    )
    assert "auc_roc" not in fallback_metrics


def test_integrated_gradients_manual_and_captum_paths() -> None:
    from src.interpretability.integrated_gradients import IntegratedGradientsGenerator

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x.flatten(1))

    model = TinyModel()
    image = torch.ones(1, 1, 2, 2)
    generator = IntegratedGradientsGenerator(model, n_steps=2, internal_batch_size=2)
    generator._ig = None
    manual = generator.generate(image)
    assert manual.shape == (2, 2)

    class FakeIG:
        def attribute(self, image, baseline, target, n_steps, internal_batch_size):
            assert target == 1
            return torch.ones_like(image)

    generator._ig = FakeIG()
    captum = generator.generate(image.squeeze(0), target_class=1)
    assert np.all(captum == 1)


def test_gradcam_internal_paths() -> None:
    from src.interpretability.gradcam import GradCAMGenerator, get_target_layer

    model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Flatten(), nn.Linear(4, 2))
    target = model[0]
    generator = GradCAMGenerator(model, target, use_captum=False)
    image = torch.ones(1, 1, 2, 2)
    heatmap = generator.generate(image)
    assert heatmap.shape == (2, 2)

    generator._captum_gc = types.SimpleNamespace(
        attribute=lambda image, target, relu_attributions: torch.ones(1, 1, 2, 2)
    )
    generator.use_captum = True
    captum_heatmap = generator.generate(image, target_class=1)
    assert captum_heatmap.shape == (2, 2)
    assert np.all(captum_heatmap == 1)

    class NoConv(nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(ValueError):
        get_target_layer(NoConv(), "unknown")


def test_transform_and_backbone_uncovered_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data.transforms import extract_2d_slice
    from src.models import backbones

    tensor_seg = torch.zeros(1, 3, 3, 3)
    tensor_seg[:, :, :, 2] = 1
    tensor_volume = torch.ones(1, 3, 3, 3)
    tensor_slice, tensor_idx = extract_2d_slice(
        tensor_volume, tensor_seg, context_slices=1
    )
    assert tensor_idx == 2
    assert tensor_slice.shape == (3, 3, 3)
    assert torch.all(tensor_slice[2] == 0)

    np_volume = np.ones((3, 3, 3), dtype=np.float32)
    np_slice, np_idx = extract_2d_slice(np_volume, None, context_slices=1)
    assert np_idx == 1
    assert np_slice.shape == (3, 3, 3)

    class FakeDenseNet(nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.kwargs = kwargs
            self.layer = nn.Linear(1, 1)

    fake_nets = types.ModuleType("monai.networks.nets")
    fake_nets.DenseNet121 = FakeDenseNet
    fake_networks = types.ModuleType("monai.networks")
    fake_networks.nets = fake_nets
    fake_monai = types.ModuleType("monai")
    fake_monai.networks = fake_networks
    monkeypatch.setitem(sys.modules, "monai", fake_monai)
    monkeypatch.setitem(sys.modules, "monai.networks", fake_networks)
    monkeypatch.setitem(sys.modules, "monai.networks.nets", fake_nets)

    dense, feature_dim = backbones.build_backbone(
        "densenet121", in_channels=2, pretrained=False
    )
    assert isinstance(dense, FakeDenseNet)
    assert feature_dim == 1024

    original = nn.Conv2d(3, 2, 1, bias=True)
    with torch.no_grad():
        original.weight.fill_(3.0)
        original.bias.fill_(2.0)
    adapted = backbones._adapt_first_conv(original, 4, pretrained=True)
    assert adapted.weight.shape[1] == 4
    assert torch.all(adapted.weight == 3.0)
    assert torch.all(adapted.bias == 2.0)


def test_validate_attention_end_to_end_batch_path() -> None:
    from src.interpretability.attention_validator import validate_attention

    class FixedModel(nn.Module):
        def eval(self):
            return self

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[0.1, 0.9], [0.8, 0.2]])

    class FixedGradCAM:
        def generate(self, image: torch.Tensor, target_class: int) -> np.ndarray:
            heatmap = np.zeros((4, 4), dtype=np.float32)
            heatmap[:2, :2] = 1.0
            return heatmap

    dataloader = [
        {
            "image": torch.ones(2, 1, 4, 4),
            "label": torch.tensor([1, 1]),
            "segmentation": torch.ones(2, 1, 4, 4),
            "patient_id": ["a", "b"],
        }
    ]

    summary = validate_attention(
        FixedModel(),
        dataloader,
        FixedGradCAM(),
        correct_only=True,
        max_samples=1,
    )
    assert summary.n_validated == 1
    assert summary.per_sample[0].patient_id == "a"


def test_visualization_functions_create_files(tmp_path: Path) -> None:
    from src.interpretability.visualization import (
        plot_gradcam_grid,
        plot_gradcam_overlay,
        plot_iou_distribution,
        plot_naive_vs_rigorous_gradcam,
    )

    image = np.ones((1, 5, 5), dtype=np.float32)
    heatmap = np.eye(5, dtype=np.float32)
    seg = np.eye(5, dtype=np.int32)

    plot_gradcam_overlay(image, heatmap, seg, tmp_path / "overlay.png", iou=0.5)
    plot_naive_vs_rigorous_gradcam(
        image, heatmap, image, heatmap, seg, 0.1, 0.7, tmp_path / "compare.png"
    )
    plot_iou_distribution([0.1, 0.4, 0.8], [0, 1, 1], {0: "LGG", 1: "HGG"}, tmp_path / "iou.png")
    plot_gradcam_grid(
        [
            {
                "image": image,
                "heatmap": heatmap,
                "segmentation": seg,
                "patient_id": "p",
                "iou": 0.6,
            }
        ],
        tmp_path / "grid.png",
        n_cols=1,
    )

    assert {p.name for p in tmp_path.glob("*.png")} == {
        "overlay.png",
        "compare.png",
        "iou.png",
        "grid.png",
    }


def test_package_lazy_imports() -> None:
    import src.data as data_pkg
    import src.interpretability as interp_pkg
    import src.models as models_pkg

    assert data_pkg.BraTSDataset.__name__ == "BraTSDataset"
    assert data_pkg.BrainTumorDataModule.__name__ == "BrainTumorDataModule"
    assert interp_pkg.GradCAMGenerator.__name__ == "GradCAMGenerator"
    assert interp_pkg.validate_attention.__name__ == "validate_attention"
    assert models_pkg.TumorClassifier.__name__ == "TumorClassifier"
    assert models_pkg.build_backbone.__name__ == "build_backbone"

    with pytest.raises(ModuleNotFoundError):
        data_pkg.__getattr__("UCSFPDGMDataset")
    with pytest.raises(AttributeError):
        data_pkg.__getattr__("does_not_exist")
    with pytest.raises(AttributeError):
        interp_pkg.__getattr__("does_not_exist")
    with pytest.raises(AttributeError):
        models_pkg.__getattr__("does_not_exist")


def test_preprocessing_ucsf_with_fake_simpleitk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.data.preprocessing import (
        apply_n4_bias_correction,
        preprocess_ucsf_patient,
        resample_volume,
    )

    class FakeImage:
        def __init__(self, data: np.ndarray) -> None:
            self.data = data

        def GetSize(self):
            return tuple(reversed(self.data.shape))

        def GetSpacing(self):
            return (2.0, 2.0, 2.0)

        def GetOrigin(self):
            return (0.0, 0.0, 0.0)

        def GetDirection(self):
            return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        def GetPixelID(self):
            return 1

        def GetDimension(self):
            return 3

        def __truediv__(self, other):
            return self

    class FakeCorrector:
        def SetMaximumNumberOfIterations(self, iterations):
            self.iterations = iterations

        def SetConvergenceThreshold(self, threshold):
            self.threshold = threshold

        def Execute(self, image, mask):
            return image

        def GetLogBiasFieldAsImage(self, image):
            return image

    fake_sitk = types.ModuleType("SimpleITK")
    fake_sitk.sitkFloat32 = 1
    fake_sitk.sitkInt32 = 2
    fake_sitk.sitkBSpline = 3
    fake_sitk.sitkNearestNeighbor = 4
    fake_sitk.sitkLinear = 5
    fake_sitk.ReadImage = lambda path, pixel_id: FakeImage(np.ones((3, 3, 3)))
    fake_sitk.GetArrayFromImage = lambda image: image.data
    fake_sitk.Transform = lambda: object()
    fake_sitk.Cast = lambda image, pixel_id: image
    fake_sitk.OtsuThreshold = lambda image, lower, upper, bins: image
    fake_sitk.Shrink = lambda image, factors: image
    fake_sitk.Exp = lambda image: image
    fake_sitk.N4BiasFieldCorrectionImageFilter = FakeCorrector
    fake_sitk.Resample = (
        lambda image, new_size, transform, interpolator, origin, spacing, direction, default, pixel_id: image
    )
    monkeypatch.setitem(sys.modules, "SimpleITK", fake_sitk)
    monkeypatch.setattr(
        "src.data.preprocessing.apply_n4_bias_correction",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("n4 failed")),
    )

    patient_id = "ucsf"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()
    for modality in ["t1", "t1ce", "t2", "flair"]:
        (patient_dir / f"{patient_id}_{modality}.nii.gz").touch()
    (patient_dir / f"{patient_id}_seg.nii.gz").touch()

    stats = preprocess_ucsf_patient(
        patient_dir, tmp_path / "out", apply_n4=True, do_resample=True
    )
    assert stats is not None
    assert stats.resampled is True
    assert stats.n4_applied is False
    assert (tmp_path / "out" / "ucsf.npz").exists()

    image = FakeImage(np.ones((2, 2, 2)))
    assert resample_volume(image, interpolation="linear") is image
    assert apply_n4_bias_correction(image, iterations=None) is image


def test_metrics_error_and_multiclass_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.evaluation import metrics as metrics_module
    from src.evaluation.metrics import compute_full_report, compute_roc_curve_data

    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    y_prob = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.6, 0.3],
            [0.7, 0.2, 0.1],
            [0.2, 0.3, 0.5],
            [0.1, 0.2, 0.7],
        ]
    )
    report = compute_full_report(y_true, y_pred, y_prob)
    assert report.auc_roc_macro > 0
    assert set(report.per_class_auc) == {0, 1, 2}

    monkeypatch.setattr(
        metrics_module, "roc_auc_score", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad auc"))
    )
    monkeypatch.setattr(
        metrics_module,
        "precision_recall_curve",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad pr")),
    )
    failed = compute_full_report(np.array([0, 1]), np.array([0, 1]), np.ones((2, 2)))
    assert failed.auc_roc_macro == 0.0
    assert failed.pr_auc_macro == 0.0

    roc = compute_roc_curve_data(np.array([0, 1]), np.array([0.1, 0.9]))
    assert roc["auc"] == 1.0


def test_bootstrap_skips_failed_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.evaluation import bootstrap as bootstrap_module
    from src.evaluation.bootstrap import bootstrap_single_metric, patient_level_bootstrap

    calls = {"n": 0}
    original = bootstrap_module.compute_full_report

    def flaky_report(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("degenerate")
        return original(*args, **kwargs)

    monkeypatch.setattr(bootstrap_module, "compute_full_report", flaky_report)
    results = patient_level_bootstrap(
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 1, 1]),
        np.array(["a", "b", "c", "d"]),
        n_iterations=3,
    )
    assert results["accuracy"].n_iterations == 2

    metric_calls = {"n": 0}

    def sometimes_bad(y_true, y_pred):
        metric_calls["n"] += 1
        if metric_calls["n"] == 2:
            raise RuntimeError("skip")
        return float(np.mean(y_true == y_pred))

    single = bootstrap_single_metric(
        np.array([0, 1]),
        np.array([0, 1]),
        np.array(["a", "b"]),
        sometimes_bad,
        n_iterations=3,
    )
    assert single.n_iterations == 2


def test_ablation_significance_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.experiments import ablation_runner as module
    from src.experiments.ablation_runner import AblationResult, AblationStudy

    baseline = AblationResult(
        variant_name="baseline",
        per_fold_metrics=[{"balanced_accuracy": 0.8}, {"balanced_accuracy": 0.9}],
    )
    variant = AblationResult(
        variant_name="short",
        per_fold_metrics=[{"balanced_accuracy": 0.7}],
    )
    study = AblationStudy(baseline=baseline, variants=[variant])
    insufficient = module._run_significance_tests(study)
    assert insufficient[0]["note"] == "Insufficient folds for testing"

    baseline.per_fold_metrics = [
        {"balanced_accuracy": 0.8},
        {"balanced_accuracy": 0.82},
        {"balanced_accuracy": 0.81},
    ]
    variant.per_fold_metrics = [
        {"balanced_accuracy": 0.7},
        {"balanced_accuracy": 0.72},
        {"balanced_accuracy": 0.71},
    ]
    monkeypatch.setattr(
        "src.evaluation.statistical_tests.wilcoxon_signed_rank_test",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    errored = module._run_significance_tests(study)
    assert errored[0]["error"] == "boom"


def test_classifier_validation_and_optimizer_paths() -> None:
    from src.models.classifier import TumorClassifier

    model = TumorClassifier(
        backbone_name="simple_cnn",
        pretrained=False,
        in_channels=4,
        head_hidden_dim=8,
        head_dropout=0.0,
        freeze_backbone=False,
    )
    model.log = lambda *args, **kwargs: None
    batch = {"image": torch.randn(2, 4, 16, 16), "label": torch.tensor([0, 1])}

    model.validation_step(batch, 0)
    model.trainer = types.SimpleNamespace(max_epochs=5)
    optim_cfg = model.configure_optimizers()
    assert len(optim_cfg["optimizer"].param_groups) == 2
    assert optim_cfg["lr_scheduler"]["monitor"] == "val/balanced_accuracy"

    frozen = TumorClassifier(
        backbone_name="simple_cnn",
        pretrained=False,
        in_channels=4,
        head_hidden_dim=8,
        head_dropout=0.0,
        freeze_backbone=True,
    )
    frozen.trainer = types.SimpleNamespace(max_epochs=5)
    frozen_cfg = frozen.configure_optimizers()
    assert len(frozen_cfg["optimizer"].param_groups) == 1


def test_set_monai_determinism_falls_back_when_hook_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.utils.reproducibility as repro

    called = {}
    monkeypatch.setattr(repro, "set_determinism", None)
    monkeypatch.setattr(repro, "set_global_seed", lambda seed: called.setdefault("seed", seed))

    repro.set_monai_determinism(123)

    assert called["seed"] == 123


def test_remaining_data_and_label_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.data as data_pkg
    from src.data.brats_dataset import BraTSDataset, MODALITY_SUFFIXES, SEG_SUFFIX
    from src.data.datamodule import BrainTumorDataModule, SliceDataset
    from src.data.label_derivation import (
        derive_brats_labels,
        derive_brats_labels_from_metadata,
        derive_brats_labels_from_segmentation,
        load_ucsf_labels,
    )

    fake_ucsf = types.ModuleType("src.data.ucsf_dataset")
    fake_ucsf.UCSFPDGMDataset = type("UCSFPDGMDataset", (), {})
    monkeypatch.setitem(sys.modules, "src.data.ucsf_dataset", fake_ucsf)
    assert data_pkg.__getattr__("UCSFPDGMDataset").__name__ == "UCSFPDGMDataset"

    fake_tio = types.ModuleType("torchio")

    class FakeScalarImage:
        def __init__(self, tensor):
            self.data = tensor

    fake_tio.ScalarImage = FakeScalarImage
    fake_tio.Subject = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "torchio", fake_tio)
    transformed_ds = SliceDataset(
        [{"image": [[1.0]], "label": 1, "patient_id": "p"}],
        transform=lambda subject: subject,
    )
    transformed_sample = transformed_ds[0]
    assert transformed_sample["image"].shape == (1, 1)
    np.testing.assert_array_equal(transformed_ds.get_labels(), np.array([1]))

    manifest_root = tmp_path / "manifest_root"
    manifest_root.mkdir()
    np.savez(manifest_root / "m.npz", image=np.ones((1, 2, 2)), label=1)
    (manifest_root / "manifest.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "file": "m.npz",
                        "label": 1,
                        "patient_id": "m",
                    }
                ]
            }
        )
    )
    dm = BrainTumorDataModule(manifest_root, tmp_path / "unused.json")
    assert dm._load_all_samples()[0]["patient_id"] == "m"
    empty_dir = tmp_path / "empty_npz"
    empty_dir.mkdir()
    assert BrainTumorDataModule(empty_dir, tmp_path / "unused.json")._load_from_directory() == []

    brats_root = tmp_path / "brats"
    brats_root.mkdir()
    (brats_root / "not_a_patient.txt").write_text("skip")
    missing_seg_id = "BraTS-missing-seg"
    missing_seg_dir = brats_root / missing_seg_id
    missing_seg_dir.mkdir()
    for suffix in MODALITY_SUFFIXES.values():
        (missing_seg_dir / f"{missing_seg_id}{suffix}").touch()
    with pytest.raises(RuntimeError):
        BraTSDataset(str(brats_root), {missing_seg_id: 1})

    valid_id = "BraTS-valid"
    valid_dir = brats_root / valid_id
    valid_dir.mkdir()
    volume = np.ones((2, 2, 5), dtype=np.float32)
    for suffix in MODALITY_SUFFIXES.values():
        _write_nifti(valid_dir / f"{valid_id}{suffix}", volume)
    seg = np.zeros((2, 2, 5), dtype=np.float32)
    seg[:, :, 4] = 1
    _write_nifti(valid_dir / f"{valid_id}{SEG_SUFFIX}", seg)
    dataset = BraTSDataset(str(brats_root), {valid_id: 0}, context_slices=0)
    assert dataset[0]["image"].shape == (4, 2, 2)
    dataset.context_slices = 2
    assert dataset._extract_slices({m: volume for m in dataset.modalities}, 0).shape == (
        20,
        2,
        2,
    )
    assert dataset._extract_slices({m: volume for m in dataset.modalities}, 4).shape == (
        20,
        2,
        2,
    )
    dataset.context_slices = 10
    assert dataset._extract_slices({m: volume for m in dataset.modalities}, 0).shape == (
        20,
        2,
        2,
    )

    metadata = tmp_path / "metadata.csv"
    metadata.write_text("ID,Grade\n,LGG\nBraTS-csv,HGG\n")
    assert derive_brats_labels_from_metadata(metadata) == {"BraTS-csv": 1}

    seg_root = tmp_path / "seg_root"
    seg_root.mkdir()
    (seg_root / "not_dir").write_text("skip")
    no_seg_dir = seg_root / "BraTS-no-seg"
    no_seg_dir.mkdir()
    hgg_dir = seg_root / "BraTS-hgg"
    hgg_dir.mkdir()
    hgg_seg = np.full((2, 2, 2), 3, dtype=np.float32)
    _write_nifti(hgg_dir / f"BraTS-hgg{SEG_SUFFIX}", hgg_seg)
    assert derive_brats_labels_from_segmentation(seg_root, enhancing_threshold=1) == {
        "BraTS-hgg": 1
    }
    (seg_root / "name_mapping.csv").write_text("ID,Grade\nBraTS-search,LGG\n")
    searched = derive_brats_labels(seg_root, metadata_path=None)
    assert searched["BraTS-search"] == 0

    no_meta_root = tmp_path / "no_meta"
    no_meta_root.mkdir()
    fallback_dir = no_meta_root / "BraTS-fallback"
    fallback_dir.mkdir()
    fallback_seg = np.zeros((1, 1, 1), dtype=np.float32)
    _write_nifti(fallback_dir / f"BraTS-fallback{SEG_SUFFIX}", fallback_seg)
    assert derive_brats_labels(no_meta_root) == {"BraTS-fallback": 0}

    ucsf_csv = tmp_path / "ucsf.csv"
    ucsf_csv.write_text("patient_id,grade\n,4\nucsf-ok,4\n")
    assert load_ucsf_labels(ucsf_csv)["ucsf-ok"]["binary_label"] == 1


def test_remaining_preprocessing_quality_and_transform_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.data.brats_dataset import MODALITY_SUFFIXES, SEG_SUFFIX
    from src.data.preprocessing import (
        preprocess_brats_patient,
        preprocess_ucsf_patient,
    )
    from src.data.quality_control import DatasetQualityControl
    from src.data.transforms import extract_2d_slice

    missing_patient = tmp_path / "missing_brats"
    missing_patient.mkdir()
    assert preprocess_brats_patient(missing_patient, tmp_path / "out") is None

    empty_patient = tmp_path / "empty_brats"
    empty_patient.mkdir()
    for suffix in MODALITY_SUFFIXES.values():
        _write_nifti(empty_patient / f"empty_brats{suffix}", np.zeros((2, 2, 2)))
    assert (
        preprocess_brats_patient(
            empty_patient, tmp_path / "out_empty", expected_shape=(2, 2, 2)
        )
        is None
    )

    no_seg_patient = tmp_path / "no_seg_brats"
    no_seg_patient.mkdir()
    for suffix in MODALITY_SUFFIXES.values():
        _write_nifti(no_seg_patient / f"no_seg_brats{suffix}", np.ones((2, 2, 2)))
    assert (
        preprocess_brats_patient(
            no_seg_patient, tmp_path / "out_no_seg", expected_shape=(2, 2, 2)
        )
        is None
    )

    edge_patient = tmp_path / "edge_brats"
    edge_patient.mkdir()
    edge_vol = np.ones((2, 2, 5), dtype=np.float32)
    for suffix in MODALITY_SUFFIXES.values():
        _write_nifti(edge_patient / f"edge_brats{suffix}", edge_vol)
    edge_seg = np.zeros((2, 2, 5), dtype=np.float32)
    edge_seg[:, :, 4] = 1
    _write_nifti(edge_patient / f"edge_brats{SEG_SUFFIX}", edge_seg)
    assert (
        preprocess_brats_patient(
            edge_patient,
            tmp_path / "out_edge",
            expected_shape=(2, 2, 5),
            context_slices=2,
        ).final_shape
        == (20, 2, 2)
    )
    assert (
        preprocess_brats_patient(
            edge_patient,
            tmp_path / "out_edge_break",
            expected_shape=(2, 2, 5),
            context_slices=10,
        ).final_shape
        == (20, 2, 2)
    )
    assert (
        preprocess_brats_patient(
            edge_patient,
            tmp_path / "out_center",
            expected_shape=(2, 2, 5),
            slice_method="center",
            context_slices=0,
        ).final_shape
        == (4, 2, 2)
    )

    class FakeImage:
        def __init__(self, data: np.ndarray) -> None:
            self.data = data

        def GetSize(self):
            return tuple(reversed(self.data.shape))

    fake_sitk = types.ModuleType("SimpleITK")
    fake_sitk.sitkFloat32 = 1
    fake_sitk.sitkInt32 = 2
    def fake_read_image(path, pixel_id):
        data = np.ones((5, 3, 3), dtype=np.float32)
        if "seg" in str(path):
            data = np.zeros((5, 3, 3), dtype=np.float32)
            data[4, :, :] = 1
        return FakeImage(data)

    fake_sitk.ReadImage = fake_read_image
    fake_sitk.GetArrayFromImage = lambda image: image.data
    monkeypatch.setitem(sys.modules, "SimpleITK", fake_sitk)
    monkeypatch.setattr("src.data.preprocessing.resample_volume", lambda image, *a, **k: image)
    monkeypatch.setattr(
        "src.data.preprocessing.apply_n4_bias_correction",
        lambda image, *a, **k: image,
    )
    ucsf_missing = tmp_path / "ucsf_missing"
    ucsf_missing.mkdir()
    assert preprocess_ucsf_patient(ucsf_missing, tmp_path / "out_ucsf_missing") is None
    ucsf_patient = tmp_path / "ucsf_success"
    ucsf_patient.mkdir()
    for modality in ["t1", "t1ce", "t2", "flair"]:
        (ucsf_patient / f"ucsf_success_{modality}.nii.gz").touch()
    ucsf_stats = preprocess_ucsf_patient(
        ucsf_patient,
        tmp_path / "out_ucsf",
        apply_n4=True,
        do_resample=False,
        context_slices=0,
    )
    assert ucsf_stats.n4_applied is True
    assert ucsf_stats.final_shape == (4, 3, 3)
    (ucsf_patient / "ucsf_success_seg.nii.gz").touch()
    edge_ucsf = preprocess_ucsf_patient(
        ucsf_patient,
        tmp_path / "out_ucsf_edge",
        apply_n4=False,
        do_resample=False,
        context_slices=2,
    )
    assert edge_ucsf.final_shape == (20, 3, 3)
    break_ucsf = preprocess_ucsf_patient(
        ucsf_patient,
        tmp_path / "out_ucsf_break",
        apply_n4=False,
        do_resample=False,
        context_slices=10,
    )
    assert break_ucsf.final_shape == (20, 3, 3)

    def fake_read_image_first_slice(path, pixel_id):
        data = np.ones((5, 3, 3), dtype=np.float32)
        if "seg" in str(path):
            data = np.zeros((5, 3, 3), dtype=np.float32)
            data[0, :, :] = 1
        return FakeImage(data)

    fake_sitk.ReadImage = fake_read_image_first_slice
    append_ucsf = preprocess_ucsf_patient(
        ucsf_patient,
        tmp_path / "out_ucsf_append",
        apply_n4=False,
        do_resample=False,
        context_slices=2,
    )
    assert append_ucsf.final_shape == (20, 3, 3)

    qc_patient = tmp_path / "qc_patient"
    qc_patient.mkdir()
    _write_nifti(qc_patient / "qc_patient_t1n.nii.gz", np.array([[[np.nan]]]))
    _write_nifti(qc_patient / "qc_patient_seg.nii.gz", np.zeros((1, 1, 1)))
    qc = DatasetQualityControl(expected_shape=None, min_tumor_voxels=2)
    qc_result = qc.check_patient(qc_patient)
    assert any("NaN/Inf" in issue for issue in qc_result.issues)
    assert any("Insufficient tumor voxels" in issue for issue in qc_result.issues)

    zero_seg = np.zeros((3, 3, 3), dtype=np.int32)
    zero_slice, zero_idx = extract_2d_slice(
        np.ones((3, 3, 3), dtype=np.float32), zero_seg
    )
    assert zero_idx == 1
    assert zero_slice.shape == (3, 3)
    direct_slice, _ = extract_2d_slice(np.ones((3, 3, 3)), method="center")
    assert direct_slice.shape == (3, 3)
    padded, _ = extract_2d_slice(np.ones((3, 3, 3)), method="center", context_slices=2)
    assert np.all(padded[0] == 0)


def test_remaining_gradcam_integrated_visualization_and_model_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.interpretability.gradcam import GradCAMGenerator, get_target_layer
    from src.interpretability.integrated_gradients import IntegratedGradientsGenerator
    from src.interpretability.visualization import (
        plot_gradcam_grid,
        plot_iou_distribution,
    )
    from src.models.baseline_svm import evaluate_baseline
    from src.models.classifier import TumorClassifier

    original_import = builtins.__import__

    def import_without_captum(name, *args, **kwargs):
        if name == "captum.attr":
            raise ImportError("captum disabled")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_captum)
    fallback_model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Flatten(), nn.Linear(4, 2))
    fallback_gc = GradCAMGenerator(fallback_model, fallback_model[0], use_captum=True)
    assert fallback_gc.use_captum is False
    fallback_ig = IntegratedGradientsGenerator(nn.Flatten())
    assert fallback_ig._ig is None
    monkeypatch.setattr(builtins, "__import__", original_import)

    class FakeLayerGradCam:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer

        def attribute(self, image, target, relu_attributions):
            return torch.ones(1, 1, 2, 2)

    fake_captum_attr = types.ModuleType("captum.attr")
    fake_captum_attr.LayerGradCam = FakeLayerGradCam
    fake_captum = types.ModuleType("captum")
    fake_captum.attr = fake_captum_attr
    monkeypatch.setitem(sys.modules, "captum", fake_captum)
    monkeypatch.setitem(sys.modules, "captum.attr", fake_captum_attr)
    captum_init_model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Flatten(), nn.Linear(4, 2))
    captum_init = GradCAMGenerator(captum_init_model, captum_init_model[0], use_captum=True)
    assert captum_init._captum_gc.target_layer is captum_init_model[0]

    class CaptumModel(nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[0.2, 0.8]])

    captum_generator = GradCAMGenerator(CaptumModel(), nn.Conv2d(1, 1, 1), use_captum=False)
    captum_generator.use_captum = True
    captum_generator._captum_gc = types.SimpleNamespace(
        attribute=lambda image, target, relu_attributions: torch.ones(1, 2, 2, 2)
    )
    assert captum_generator.generate(torch.ones(1, 1, 2, 2)).shape == (2, 2)

    no_hook_generator = GradCAMGenerator(
        nn.Sequential(nn.Flatten(), nn.Linear(4, 2)),
        nn.Conv2d(1, 1, 1),
        use_captum=False,
    )
    np.testing.assert_array_equal(
        no_hook_generator.generate(torch.ones(1, 1, 2, 2)),
        np.zeros((2, 2)),
    )
    resized = GradCAMGenerator._resize_heatmap(np.ones((1, 1)), 2, 2)
    assert resized.shape == (2, 2)
    np.testing.assert_array_equal(
        GradCAMGenerator._normalize_heatmap(np.zeros((2, 2))), np.zeros((2, 2))
    )

    assert get_target_layer(types.SimpleNamespace(backbone=types.SimpleNamespace(features=nn.Sequential())), "densenet") is not None
    assert isinstance(
        get_target_layer(nn.Sequential(nn.Sequential(nn.Conv2d(1, 1, 1))), "densenet"),
        nn.Conv2d,
    )
    assert isinstance(
        get_target_layer(nn.Sequential(nn.BatchNorm2d(1)), "densenet"),
        nn.BatchNorm2d,
    )
    assert get_target_layer(types.SimpleNamespace(conv_head=nn.Conv2d(1, 1, 1)), "efficientnet").out_channels == 1
    assert get_target_layer(types.SimpleNamespace(blocks=[nn.Conv2d(1, 1, 1)]), "efficientnet").out_channels == 1

    class FakeIG:
        def attribute(self, image, baseline, target, n_steps, internal_batch_size):
            return torch.ones(1, 2, 2, 2)

    ig_generator = IntegratedGradientsGenerator(nn.Flatten())
    ig_generator._ig = FakeIG()
    assert ig_generator.generate(torch.ones(1, 2, 2, 2), target_class=0).shape == (2, 2)
    manual_ig = IntegratedGradientsGenerator(nn.Sequential(nn.Flatten(), nn.Linear(8, 2)), n_steps=2)
    manual_ig._ig = None
    assert manual_ig.generate(torch.ones(1, 2, 2, 2), target_class=0).shape == (2, 2)

    plot_iou_distribution([0.1, 0.2], output_path=tmp_path / "iou_plain.png")
    plot_gradcam_grid(
        [
            {"image": np.ones((1, 3, 3)), "heatmap": np.ones((3, 3)), "iou": 0.4},
        ],
        tmp_path / "grid_empty_axis.png",
        n_cols=2,
    )
    assert (tmp_path / "iou_plain.png").exists()
    assert (tmp_path / "grid_empty_axis.png").exists()

    class MultiProbaModel:
        def predict(self, x):
            return np.array([0, 1, 2])

        def predict_proba(self, x):
            return np.eye(3)

    assert "auc_roc" in evaluate_baseline(
        MultiProbaModel(), np.zeros((3, 1)), np.array([0, 1, 2]), "multi"
    )

    model = TumorClassifier(
        backbone_name="simple_cnn",
        pretrained=False,
        in_channels=4,
        head_hidden_dim=8,
        head_dropout=0.0,
    )
    model.backbone = nn.Sequential()
    model.backbone.forward = lambda x: (torch.ones(x.shape[0], 8, 2, 2),)
    model.head = nn.Linear(8, 2)
    assert model(torch.ones(2, 4, 4, 4)).shape == (2, 2)


def test_remaining_evaluation_experiment_logging_and_repro_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omegaconf import OmegaConf

    from src.evaluation import bootstrap as bootstrap_module
    from src.evaluation import statistical_tests as stats_module
    from src.evaluation.bootstrap import patient_level_bootstrap
    from src.evaluation.statistical_tests import wilcoxon_signed_rank_test
    from src.experiments import ablation_runner as ablation_module
    from src.experiments.ablation_runner import (
        AblationResult,
        AblationStudy,
        AblationVariant,
        run_ablation_study,
    )
    from src.experiments.figure_generator import plot_confusion_matrix
    from src.interpretability.attention_validator import compute_dice, validate_attention
    from src.utils import logging_utils
    import src.utils.reproducibility as repro

    full_report = bootstrap_module.compute_full_report
    calls = {"n": 0}

    def full_then_fail(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("skip all bootstrap iterations")
        return full_report(*args, **kwargs)

    monkeypatch.setattr(bootstrap_module, "compute_full_report", full_then_fail)
    empty_boot = patient_level_bootstrap(
        np.array([0, 1]),
        np.array([0, 1]),
        np.array(["a", "b"]),
        n_iterations=2,
    )
    assert empty_boot == {}

    monkeypatch.setattr(
        stats_module.stats,
        "wilcoxon",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad")),
    )
    wilcoxon_result = wilcoxon_signed_rank_test([0.9, 0.8, 0.7], [0.7, 0.6, 0.5])
    assert wilcoxon_result.p_value == 1.0

    study = AblationStudy(
        study_name="Coverage",
        baseline=AblationResult(
            variant_name="baseline",
            metrics={"balanced_accuracy": 0.8},
        ),
        variants=[
            AblationResult(
                variant_name="variant",
                metrics={"balanced_accuracy": 0.7},
                delta_from_baseline={"balanced_accuracy": -0.1},
            )
        ],
        statistical_tests=[{"variant": "variant", "significant": True}],
    )
    assert "$p < 0.05$" in study.generate_latex_table()

    class FakeTest:
        def to_dict(self):
            return {"test": "fake", "significant": False}

    monkeypatch.setattr(
        "src.evaluation.statistical_tests.wilcoxon_signed_rank_test",
        lambda *args, **kwargs: FakeTest(),
    )
    sig_study = AblationStudy(
        baseline=AblationResult(
            per_fold_metrics=[
                {"balanced_accuracy": 0.8},
                {"balanced_accuracy": 0.81},
                {"balanced_accuracy": 0.82},
            ]
        ),
        variants=[
            AblationResult(
                variant_name="sig",
                per_fold_metrics=[
                    {"balanced_accuracy": 0.7},
                    {"balanced_accuracy": 0.71},
                    {"balanced_accuracy": 0.72},
                ],
            )
        ],
    )
    assert ablation_module._run_significance_tests(sig_study)[0]["test"] == "fake"

    def fake_train(config):
        return AblationResult(
            metrics={"balanced_accuracy": config["score"]},
            per_fold_metrics=[
                {"balanced_accuracy": config["score"]},
                {"balanced_accuracy": config["score"]},
                {"balanced_accuracy": config["score"]},
            ],
        )

    no_sig = run_ablation_study(
        "No Significance",
        "question",
        {"score": 0.8},
        [AblationVariant("v", "desc", config_overrides={"score": 0.7})],
        fake_train,
        run_significance_tests=False,
    )
    assert no_sig.statistical_tests == []
    with_sig = run_ablation_study(
        "With Significance",
        "question",
        {"score": 0.8},
        [AblationVariant("v", "desc", config_overrides={"score": 0.7})],
        fake_train,
        run_significance_tests=True,
    )
    assert with_sig.statistical_tests[0]["test"] == "fake"

    plot_confusion_matrix(
        np.array([[1, 2], [3, 4]]),
        ["A", "B"],
        tmp_path / "cm_counts.png",
        normalize=False,
    )
    assert (tmp_path / "cm_counts.png").exists()

    assert compute_dice(np.zeros((2, 2)), np.zeros((2, 2))) == 0.0

    class FixedModel(nn.Module):
        def eval(self):
            return self

        def forward(self, images):
            return torch.tensor([[0.9, 0.1]])

    class ShouldNotRunGradCAM:
        def generate(self, *args, **kwargs):
            raise AssertionError("incorrect sample should be skipped")

    skipped = validate_attention(
        FixedModel(),
        [
            {
                "image": torch.ones(1, 1, 2, 2),
                "label": torch.tensor([1]),
                "segmentation": torch.ones(1, 2, 2),
            }
        ],
        ShouldNotRunGradCAM(),
        correct_only=True,
    )
    assert skipped.n_validated == 0

    cfg = OmegaConf.create(
        {
            "wandb": {"mode": "online", "project": "p", "tags": []},
            "model": {"name": "m"},
            "dataset": {"name": "d"},
            "training": {"name": "t"},
            "paths": {"logs": str(tmp_path)},
        }
    )
    fake_loggers = types.ModuleType("pytorch_lightning.loggers")

    class CSVLogger:
        def __init__(self, save_dir, name):
            self.save_dir = save_dir
            self.name = name

    fake_loggers.CSVLogger = CSVLogger
    monkeypatch.setitem(sys.modules, "pytorch_lightning.loggers", fake_loggers)
    assert logging_utils.init_wandb_logger(cfg).__class__.__name__ == "CSVLogger"

    bad_wandb = types.SimpleNamespace(
        run=types.SimpleNamespace(id="run", log_artifact=lambda artifact: None),
        Artifact=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    monkeypatch.setitem(sys.modules, "wandb", bad_wandb)
    logging_utils.log_config_as_artifact(object(), cfg)

    called = {}

    def raise_import_error(seed):
        raise ImportError("monai failed")

    monkeypatch.setattr(repro, "set_determinism", raise_import_error)
    monkeypatch.setattr(repro, "set_global_seed", lambda seed: called.setdefault("seed", seed))
    repro.set_monai_determinism(321)
    assert called["seed"] == 321
