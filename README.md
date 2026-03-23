# Brain Tumor MRI Classification

**Rigorous glioma grading from MRI: exposing and correcting artificial accuracy on Kaggle datasets.**

Author: Francisco Javier Mercader Martínez  
Institution: Universidad Politécnica de Cartagena (UPCT)  
Context: TFG/TFM — 2026

---

## Why This Project Exists

The 95–99% accuracy routinely reported on Kaggle brain tumor datasets is almost certainly artificial. Models achieve these numbers by learning scanner artifacts, dataset source signatures, and background patterns — not tumor morphology. This project proves it and corrects it.

Switching from image-level to patient-level data splitting drops reported accuracy by 7–30 percentage points, but produces results that actually generalize to unseen clinical data.

### The Thesis in One Sentence

> Methodological rigor matters more than accuracy numbers, and a properly evaluated 87% tells you more than a misleadingly reported 99%.

---

## Quick Start

```bash
git clone https://github.com/<your-username>/brain_tumor_classification_v2.git
cd brain_tumor_classification_v2
pip install -e ".[dev]"
make test                    # 168 tests pass
make download-brats          # Requires Synapse account
make all                     # Full pipeline: preprocess → train → evaluate
```

See [docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md) for detailed instructions.

---

## Project Structure

```
brain_tumor_classification_v2/
├── configs/          # Hydra YAML configs (data, model, training, experiment)
├── src/              # Source code (data, models, training, evaluation, experiments, interpretability, utils)
├── scripts/          # 10 CLI entry points
├── notebooks/        # 4 exploration scripts
├── tests/            # 168 automated tests across 6 subpackages
├── docs/             # Comprehensive documentation
├── dvc.yaml          # Reproducible DVC pipeline
├── Makefile          # 18 convenience commands
├── Dockerfile        # Container environment
└── pyproject.toml    # Pinned dependencies
```

---

## Documentation

| Document                                        | Description                                               |
| ----------------------------------------------- | --------------------------------------------------------- |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md)         | Technical architecture, module map, design decisions      |
| [DATA_GUIDE.md](docs/DATA_GUIDE.md)             | Data acquisition, preprocessing pipeline, quality control |
| [EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md) | How to run every experiment and ablation                  |
| [METHODOLOGY.md](docs/METHODOLOGY.md)           | Methodological decisions with literature citations        |
| [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)   | Complete environment and reproduction instructions        |
| [API_REFERENCE.md](docs/API_REFERENCE.md)       | Module-by-module function and class reference             |
| [CLAIM_CHECKLIST.md](docs/CLAIM_CHECKLIST.md)   | CLAIM/TRIPOD+AI compliance checklist                      |
| [CHANGELOG.md](docs/CHANGELOG.md)               | Development history across all phases                     |

---

## Expected Results

| Task                            | Metric            | Expected Range |
| ------------------------------- | ----------------- | -------------- |
| HGG vs LGG (binary)             | AUC-ROC           | 0.88 – 0.95    |
| HGG vs LGG (binary)             | Balanced Accuracy | 85% – 92%      |
| External validation (UCSF-PDGM) | Drop from BraTS   | 3 – 8 pp       |

**Results of 99% would indicate a problem, not a success.**

---

## License

MIT
