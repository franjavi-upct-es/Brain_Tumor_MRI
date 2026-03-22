# src/experiments/__init__.py — Experiment orchestration subpackage
"""
Experiment modules that implement the ablation studies and comparative
experiments.

Each experiment answers a specific methodological question:
  - split_comparison: How much does image-level splitting inflate accuracy?
  - ablation_runner: What is the contribution of each preprocessing step?
  - naive_experiment: Reproducing the flawed Kaggle approach (Chapter 3).
  - figure_generator: Publication-ready figures at 300 DPI.
"""
