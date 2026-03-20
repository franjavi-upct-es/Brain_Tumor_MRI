# Makefile — Common commands for brain tumor classification

.PHONY: help install test lint clean repro download preprocess train evaluate all

PYTHON := uv run python
PYTEST := uv run pytest
PIP := uv pip

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------- Setup ----------

install:  ## Install all dependencies
	$(PIP) install -e ".[dev]" --break-system-packages

install-prod:  ## Install production dependencies only
	$(PIP) install -e . --break-system-packages

# ---------- Quality ----------

test:  ## Run all tests (pytest)
	$(PYTEST) tests/ -v --tb=short

test-leakage:  ## Run ONLY the critical data leakage tests
	$(PYTEST) tests/test_data/test_no_leakage.py -v

test-cov:  ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint:  ## Run linter (ruff)
	ruff check src/ tests/ scripts/

lint-fix:  ## Run linter with auto-fix
	ruff check --fix src/ tests/ scripts/

# ---------- Data ----------

download:  ## Download all datasets (BraTS 2023 + UCSF-PDGM)
	$(PYTHON) scripts/download_data.py --dataset all

download-brats:  ## Download BraTS 2023 only
	$(PYTHON) scripts/download_data.py --dataset brats2023

download-ucsf:  ## Download UCSF-PDGM only
	$(PYTHON) scripts/download_data.py --dataset ucsf_pdgm

preprocess:  ## Run preprocessing pipeline
	$(PYTHON) scripts/preprocess.py --config configs/data/brats2023.yaml
	$(PYTHON) scripts/preprocess.py --config configs/data/ucsf_pdgm.yaml

splits:  ## Create patient-level cross-validation splits
	$(PYTHON) scripts/create_splits.py --seed 42

# ---------- Training ----------

train:  ## Train all 5 folds (DenseNet-121 baseline)
	@for fold in 0 1 2 3 4; do \
		echo "=== Training fold $$fold ===" ; \
		$(PYTHON) scripts/train.py model=densenet121_2d data=brats2023 training=baseline fold=$$fold ; \
	done

train-fold:  ## Train a single fold (usage: make train-fold FOLD=0)
	$(PYTHON) scripts/train.py model=densenet121_2d data=brats2023 training=baseline fold=$(FOLD)

# ---------- Evaluation ----------

evaluate:  ## Run full evaluation with bootstrap CI
	$(PYTHON) scripts/evaluate.py --all-folds --bootstrap 1000

evaluate-external:  ## Evaluate on UCSF-PDGM external dataset
	$(PYTHON) scripts/evaluate.py --dataset ucsf_pdgm --bootstrap 1000

figures:  ## Generate all thesis figures
	$(PYTHON) scripts/generate_figures.py --gradcam --iou-validation

# ---------- Reproducibility ----------

repro:  ## Reproduce entire pipeline via DVC
	dvc repro

metrics:  ## Show all tracked metrics
	dvc metrics show

# ---------- Full pipeline ----------

all: install test download preprocess splits train evaluate evaluate-external figures  ## Run everything end-to-end

# ---------- Cleanup ----------

clean:  ## Remove generated outputs (keeps raw data)
	rm -rf outputs/models/* outputs/metrics/* outputs/evaluation/* outputs/figures/*
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean  ## Remove everything including processed data
	rm -rf data/processed/* data/splits/*
