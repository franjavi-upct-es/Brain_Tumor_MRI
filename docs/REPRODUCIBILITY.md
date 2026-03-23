# Reproducibility Guide

This project is designed for complete reproducibility. Every random process
is seeded, every dependency is pinned, every pipeline step is tracked by DVC.

---

## Environment Setup

### Option A: pip (recommended for development)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Option B: Docker (guaranteed reproducibility)

```bash
docker build -t brain-tumor-v2 .
docker run --gpus all -v $(pwd)/data:/app/data brain-tumor-v2 make test
```

### Option C: Conda

```bash
conda create -n brain-tumor-v2 python=3.10
conda activate brain-tumor-v2
pip install -e ".[dev]"
```

---

## Seed Management

All random processes use seed=42 (configurable in `configs/config.yaml`):

| Library | Seeding mechanism |
|---------|-------------------|
| Python `random` | `random.seed(42)` |
| NumPy | `np.random.seed(42)` |
| PyTorch CPU | `torch.manual_seed(42)` |
| PyTorch CUDA | `torch.cuda.manual_seed_all(42)` |
| MONAI | `monai.utils.set_determinism(seed=42)` |
| cuDNN | `torch.backends.cudnn.deterministic = True` |
| Hash seed | `PYTHONHASHSEED=42` |
| DataLoader workers | `worker_init_fn=seed_worker` |
| sklearn | `random_state=42` in all splitters |
| Bootstrap | `np.random.RandomState(42)` |

**Verification:** `tests/test_models/test_determinism.py` confirms same seed produces identical weights, outputs, and training losses.

---

## DVC Pipeline

The `dvc.yaml` file defines the complete pipeline with explicit dependencies:

```bash
dvc repro           # Run the full pipeline
dvc status          # Check what needs re-running
dvc metrics show    # Display all metrics
dvc metrics diff    # Compare metrics between commits
```

Each DVC stage specifies:
- `cmd`: the exact command to run
- `deps`: input files (triggers re-run if changed)
- `outs`: output files (cached by DVC)
- `params`: config values (triggers re-run if changed)
- `metrics`: JSON files tracked across commits

---

## Verifying Reproducibility

### Step 1: Run tests

```bash
make test
# Expected: 168 passed
```

### Step 2: Check determinism

```bash
PYTHONPATH=. pytest tests/test_models/test_determinism.py -v
# All 4 tests must pass
```

### Step 3: Verify splits

```bash
PYTHONPATH=. pytest tests/test_data/test_no_leakage.py -v
# All 12 tests must pass
```

### Step 4: Run pipeline twice

```bash
dvc repro
# Record metrics
dvc metrics show > run1_metrics.txt

# Clean and re-run
make clean
dvc repro
dvc metrics show > run2_metrics.txt

# Compare — should be identical
diff run1_metrics.txt run2_metrics.txt
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | None (CPU works) | NVIDIA GPU with 8+ GB VRAM |
| RAM | 16 GB | 32 GB |
| Disk | 50 GB (BraTS only) | 250 GB (both datasets) |
| CUDA | 11.8+ | 12.x |

Training time estimates (DenseNet-121, 5-fold CV):
- GPU (RTX 3080): ~2-4 hours
- CPU only: ~24-48 hours

---

## File Versioning

| What | Tool | Storage |
|------|------|---------|
| Code | Git | GitHub/GitLab |
| Data | DVC | Local or remote (S3, GCS) |
| Configs | Git | With code |
| Models | DVC | With data |
| Metrics | Git (via DVC) | As JSON |
