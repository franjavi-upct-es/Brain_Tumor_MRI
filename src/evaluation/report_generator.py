# src/evaluation/report_generator.py
# Automated CLAIM-compliant report generation
"""
Generates publication-ready evaluation tables and summaries following
CLAIM (Tejani et al., 2024) and TRIPOD+AI (Collins et al., 2024) guidelines.

Output formats:
    - JSON for programmatic consumption and DVC tracking.
    - LaTeX tables for direct thesis inclusion.
    - Plain text for console logging.

All tables include 95% confidence intervals from bootstrap estimation.

References:
    - Tejani et al. (2024). CLAIM checklist. Radiology: AI.
    - Collins et al. (2024). TRIPOD+AI statement. BMJ.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.bootstrap import BootstrapResult
from src.evaluation.metrics import ClassificationReport

logger = logging.getLogger(__name__)


def generate_main_results_table(
    results: dict[str, dict[str, BootstrapResult]],
    class_names: dict[int, str] | None = None,
) -> str:
    """
    Generate the main results table in LaTeX format.

    Produces a table with one row per model and columns for each primary
    metric, each formatted as 'value (CI_lower — CI_upper)'.

    Parameters
    ----------
    results : dict
        Mapping from model_name to dict of metric_name → BootstrapResult.
    class_names : dict, optional
        Mapping from class index to display name.

    Returns
    -------
    str
        LaTeX table string.
    """
    metrics_order = [
        "balanced_accuracy",
        "f1_macro",
        "auc_roc_macro",
        "cohens_kappa",
    ]
    metric_labels = {
        "balanced_accuracy": "Bal. Acc.",
        "f1_macro": "F1 (macro)",
        "auc_roc_macro": "AUC-ROC",
        "cohens_kappa": "Cohen's $\\kappa$",
    }

    # Header
    col_headers = " & ".join([metric_labels.get(m, m) for m in metrics_order])
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Classification performance with 95\\% bootstrap confidence"
        " intervals (1,000 patient-level iterations).}",
        "\\label{tab:main_results}",
        f"\\begin{{tabular}}{{l{'c' * len(metrics_order)}}}",
        "\\toprule",
        f"Model & {col_headers} \\\\",
        "\\midrule",
    ]

    # Data rows
    for model_name, model_results in results.items():
        row_values = []
        for metric in metrics_order:
            if metric in model_results:
                br = model_results[metric]
                cell = (
                    f"{br.point_estimate:.3f} "
                    f"({br.ci_lower:.3f}--{br.ci_upper:.3f})"
                )
            else:
                cell = "---"
            row_values.append(cell)

        lines.append(f"{model_name} & {' & '.join(row_values)} \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def generate_confusion_matrix_latex(
    cm_absolute: np.ndarray,
    cm_normalized: np.ndarray,
    class_names: list[str],
    model_name: str = "Model",
) -> str:
    """
    Generate a confusion matrix table in LaTeX format.

    Shows absolute counts with row-normalized percentages in parentheses.

    Parameters
    ----------
    cm_absolute : np.ndarray
        Confusion matrix with raw counts.
    cm_normalized : np.ndarray
        Row-normalized confusion matrix (percentages).
    class_names : list[str]
        Class display names.
    model_name : str
        Model name for caption.

    Returns
    -------
    str
        LaTeX table string.
    """
    n_classes = len(class_names)
    col_spec = "l" + "c" * n_classes

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Confusion matrix for {model_name}. "
        "Values show count (row \\%).}}",
        f"\\label{{tab:cm_{model_name.lower().replace(' ', '_')}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "& " + " & ".join([f"Pred. {c}" for c in class_names]) + " \\\\",
        "\\midrule",
    ]

    for i, class_name in enumerate(class_names):
        cells = []
        for j in range(n_classes):
            count = int(cm_absolute[i, j])
            pct = cm_normalized[i, j]
            cells.append(f"{count} ({pct:.1f}\\%)")
        lines.append(f"True {class_name} & {' & '.join(cells)} \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def generate_comparison_table(
    naive_results: dict[str, BootstrapResult],
    rigorous_results: dict[str, BootstrapResult],
    test_results: list[dict[str, Any]] | None = None,
) -> str:
    """
    Generate naive vs rigorous comparison table.

    This is the central thesis table showing the impact of methodology.

    Parameters
    ----------
    naive_results : dict
        Bootstrap results from the naive approach.
    rigorous_results : dict
        Bootstrap results from the rigorous approach.
    test_results : list, optional
        Statistical test results for significance annotations.

    Returns
    -------
    str
        LaTeX table string.
    """
    metrics_order = [
        "balanced_accuracy",
        "f1_macro",
        "auc_roc_macro",
        "cohens_kappa",
    ]
    metric_labels = {
        "balanced_accuracy": "Balanced Accuracy",
        "f1_macro": "Macro F1",
        "auc_roc_macro": "AUC-ROC",
        "cohens_kappa": "Cohen's $\\kappa$",
    }

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Comparison of naive (Kaggle) "
        "vs rigorous (BraTS) methodology."
        " All metrics with 95\\% CI.}",
        "\\label{tab:naive_vs_rigorous}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        'Metric & Na\\"ive Approach & Rigorous Approach & $\\Delta$ \\\\',
        "\\midrule",
    ]

    for metric in metrics_order:
        label = metric_labels.get(metric, metric)
        naive_str = "---"
        rigorous_str = "---"
        delta_str = "---"

        if metric in naive_results:
            nr = naive_results[metric]
            naive_str = (
                f"{nr.point_estimate:.3f} "
                f"({nr.ci_lower:.3f}--{nr.ci_upper:.3f})"
            )

        if metric in rigorous_results:
            rr = rigorous_results[metric]
            rigorous_str = (
                f"{rr.point_estimate:.3f} "
                f"({rr.ci_lower:.3f}--{rr.ci_upper:.3f})"
            )

        if metric in naive_results and metric in rigorous_results:
            delta = (
                rigorous_results[metric].point_estimate
                - naive_results[metric].point_estimate
            )
            delta_str = f"{delta:+.3f}"

        lines.append(
            f"{label} & {naive_str} & {rigorous_str} & {delta_str} \\\\"
        )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def save_full_report(
    report: ClassificationReport,
    bootstrap_results: dict[str, BootstrapResult],
    output_dir: Path,
    model_name: str = "model",
    dataset_name: str = "dataset",
) -> None:
    """
    Save complete evaluation report as JSON.

    Parameters
    ----------
    report : ClassificationReport
        Full metrics report.
    bootstrap_results : dict
        Bootstrap CI results for all metrics.
    output_dir : Path
        Output directory.
    model_name : str
        Model identifier.
    dataset_name : str
        Dataset identifier.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_report = {
        "model": model_name,
        "dataset": dataset_name,
        "metrics": report.to_dict(),
        "bootstrap_ci": {
            name: res.to_dict() for name, res in bootstrap_results.items()
        },
    }

    path = output_dir / f"{model_name}_{dataset_name}_report.json"
    with open(path, "w") as f:
        json.dump(full_report, f, indent=2)

    logger.info("Full report saved to %s", path)


def format_metric_with_ci(result: BootstrapResult, decimals: int = 3) -> str:
    """
    Format a metric as 'value (lower — upper)' for text output.

    Parameters
    ----------
    result : BootstrapResult
        Bootstrap result to format.
    decimals : int
        Number of decimal places.

    Returns
    -------
    str
        Formatted string.
    """
    fmt = f"{{:.{decimals}f}}"
    val = fmt.format(result.point_estimate)
    lo = fmt.format(result.ci_lower)
    hi = fmt.format(result.ci_upper)
    return f"{val} ({lo} — {hi})"
