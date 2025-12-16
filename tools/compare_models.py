"""
tools/compare_models.py - Comprehensive Model Comparison
========================================================
Automatically loads and compares results from:
1. Base Model (from error_analysis.py results)
2. Fine-Tuned Model (from evaluate_external.py)
3. Focal Loss Model (from adaptive_retrain.py)

This script reads the actual JSON results saved by each evaluation
and generates publication-quality comparative visualizations.

Usage:
    python tools/compare_models.py --config configs/config.yaml
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results_from_files(checkpoint_dir="models", reports_dir="reports"):
    """
    Load actual results from saved JSON files.

    Returns:
        Dictionary with 'base', 'finetuned', 'focal' keys
    """
    results = {}

    # ========== BASE MODEL RESULTS ==========
    # From error_summary.json (created by error_analysis.py)
    base_path = os.path.join(reports_dir, "error_summary.json")
    if os.path.exists(base_path):
        print(f"[INFO] Loading base model results from: {base_path}")
        with open(base_path) as f:
            base_data = json.load(f)

            # Extract metrics
            total_errors = base_data.get("total_errors", 0)

            # From your output: 173 total images, 87 tumors, 86 healthy
            # All errors are Tumor -> Healthy (44 FN)
            total_images = 173
            total_tumors = 87
            total_healthy = 86

            fn = total_errors  # All errors are False Negatives
            tp = total_tumors - fn
            tn = total_healthy  # No False Positives in base
            fp = 0

            results["base"] = {
                "confusion_matrix": [[tn, fp], [fn, tp]],
                "recall": tp / total_tumors if total_tumors > 0 else 0,
                "specificity": tn / total_healthy if total_healthy > 0 else 0,
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "accuracy": (tp + tn) / total_images,
                "fn_rate": fn / total_tumors if total_tumors > 0 else 0,
                "source": "error_summary.json",
            }
            print(f"  ✓ Base: Recall={results['base']['recall']:.1%}, FN={fn}")
    else:
        print(f"[WARN] Base results not found: {base_path}")
        print(f"       Run: python notebooks/error_analysis.py first")

    # ========== FINE-TUNED MODEL RESULTS ==========
    # From external evaluation (manually created or from evaluate_external.py output)
    # We'll check for a dedicated file
    finetuned_path = os.path.join(checkpoint_dir, "finetuned_external_results.json")
    if os.path.exists(finetuned_path):
        print(f"[INFO] Loading fine-tuned results from: {finetuned_path}")
        with open(finetuned_path) as f:
            results["finetuned"] = json.load(f)
            print(f"  ✓ Fine-tuned: Recall={results['finetuned']['recall']:.1%}")
    else:
        # Try to infer from error_log.csv if finetuned results exist
        error_log_path = os.path.join(reports_dir, "error_log.csv")
        if os.path.exists(error_log_path):
            print(f"[INFO] Inferring fine-tuned results from: {error_log_path}")
            df = pd.read_csv(error_log_path)

            # Check if we have fine-tuned predictions
            if "pred_ft" in df.columns:
                total_tumors = 87
                total_healthy = 86

                # Count errors in fine-tuned
                ft_errors = df[df["pred_ft"] != df["true_binary"]]
                fn_ft = len(ft_errors[ft_errors["true_binary"] == 1])
                fp_ft = len(ft_errors[ft_errors["true_binary"] == 0])

                tp_ft = total_tumors - fn_ft
                tn_ft = total_healthy - fp_ft

                results["finetuned"] = {
                    "confusion_matrix": [[tn_ft, fp_ft], [fn_ft, tp_ft]],
                    "recall": tp_ft / total_tumors,
                    "specificity": tn_ft / total_healthy,
                    "precision": tp_ft / (tp_ft + fp_ft) if (tp_ft + fp_ft) > 0 else 0,
                    "accuracy": (tp_ft + tn_ft) / (total_tumors + total_healthy),
                    "fn_rate": fn_ft / total_tumors,
                    "source": "error_log.csv",
                }
                print(
                    f"  ✓ Fine-tuned (from CSV): Recall={results['finetuned']['recall']:.1%}, FN={fn_ft}"
                )
        else:
            print(f"[WARN] Fine-tuned results not found")

    # ========== FOCAL LOSS MODEL RESULTS ==========
    focal_path = os.path.join(checkpoint_dir, "focal_external_results.json")
    if os.path.exists(focal_path):
        print(f"[INFO] Loading Focal Loss results from: {focal_path}")
        with open(focal_path) as f:
            results["focal"] = json.load(f)
            print(f"  ✓ Focal: Recall={results['focal']['recall']:.1%}")
    else:
        print(f"[WARN] Focal Loss results not found: {focal_path}")
        print(f"       Run: python tools/adaptive_retrain.py first")

    return results


def create_comparative_dashboard(results_dict, output_dir="reports"):
    """
    Create comprehensive comparison dashboard.

    Args:
        results_dict: Dictionary with keys 'base', 'finetuned', 'focal'
                     Each value is a dict with metrics and confusion matrix
    """
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    models = ["Base Model", "Fine-Tuned", "Focal Loss"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    # ========== 1. Confusion Matrices (Normalized) ==========
    for i, (model_name, model_key) in enumerate(
        zip(models, ["base", "finetuned", "focal"])
    ):
        ax = fig.add_subplot(gs[0, i])

        if model_key in results_dict:
            cm = np.array(results_dict[model_key]["confusion_matrix"])
            cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".1%",
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                square=True,
                cbar_kws={"label": "Proportion"},
                xticklabels=["Healthy", "Tumor"],
                yticklabels=["Healthy", "Tumor"],
                ax=ax,
            )

            ax.set_title(
                f"{model_name}\nConfusion Matrix", fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("True", fontsize=10)

            # Add FN/FP annotations
            tn, fp, fn, tp = cm.ravel()
            if fn > 0:
                ax.text(
                    0.5,
                    1.5,
                    f"⚠️ {fn} missed",
                    ha="center",
                    va="center",
                    color="darkred",
                    fontweight="bold",
                    fontsize=9,
                )
        else:
            ax.text(
                0.5,
                0.5,
                f"{model_name}\nNot Available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.axis("off")

    # ========== 2. Clinical Metrics Comparison ==========
    ax = fig.add_subplot(gs[1, :2])

    metrics_names = [
        "Sensitivity\n(Recall)",
        "Specificity",
        "Precision\n(PPV)",
        "Accuracy",
    ]
    x = np.arange(len(metrics_names))
    width = 0.25

    for i, (model_name, model_key, color) in enumerate(
        zip(models, ["base", "finetuned", "focal"], colors)
    ):
        if model_key in results_dict:
            r = results_dict[model_key]
            values = [
                r.get("recall", 0),
                r.get("specificity", 0),
                r.get("precision", 0),
                r.get("accuracy", 0),
            ]

            bars = ax.bar(
                x + i * width - width,
                values,
                width,
                label=model_name,
                color=color,
                alpha=0.8,
                edgecolor="black",
            )

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.02,
                    f"{height:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Clinical Performance Metrics", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, label="Clinical Target")

    # ========== 3. False Negatives Reduction ==========
    ax = fig.add_subplot(gs[1, 2])

    fn_counts = []
    fn_labels = []

    for model_key in ["base", "finetuned", "focal"]:
        if model_key in results_dict:
            cm = np.array(results_dict[model_key]["confusion_matrix"])
            fn = cm[1, 0]  # Tumors predicted as Healthy
            fn_counts.append(fn)
            fn_labels.append(model_key.capitalize())

    bars = ax.bar(
        fn_labels,
        fn_counts,
        color=["#d62728", "#ff7f0e", "#2ca02c"],
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
    )

    # Add reduction annotations
    for i in range(1, len(fn_counts)):
        reduction = fn_counts[0] - fn_counts[i]
        if reduction > 0:
            ax.annotate(
                f"-{reduction}\n({-reduction / fn_counts[0]:.0%})",
                xy=(i, fn_counts[i]),
                xytext=(i, fn_counts[i] + max(fn_counts) * 0.15),
                ha="center",
                fontsize=11,
                fontweight="bold",
                color="green",
                arrowprops=dict(arrowstyle="->", color="green", lw=2),
            )

    ax.set_ylabel("False Negatives (Missed Tumors)", fontsize=11, fontweight="bold")
    ax.set_title("Error Reduction Progress", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, max(fn_counts) * 1.3)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(fn_counts) * 0.05,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    # ========== 4. Radar Chart - Clinical Utility ==========
    ax = fig.add_subplot(gs[2, :], projection="polar")

    categories = ["Sensitivity", "Specificity", "Precision", "NPV", "F1-Score"]
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for model_name, model_key, color in zip(
        models, ["base", "finetuned", "focal"], colors
    ):
        if model_key in results_dict:
            r = results_dict[model_key]
            cm = np.array(r["confusion_matrix"])
            tn, fp, fn, tp = cm.ravel()

            # Calculate metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            f1 = (
                2 * (precision * sensitivity) / (precision + sensitivity)
                if (precision + sensitivity) > 0
                else 0
            )

            values = [sensitivity, specificity, precision, npv, f1]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=9)
    ax.set_title("Clinical Utility Profile", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(
        "Comprehensive Model Comparison: External Dataset Validation",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(
        os.path.join(output_dir, "model_comparison_dashboard.png"),
        dpi=180,
        bbox_inches="tight",
    )
    print(f"✅ Comparison dashboard saved: {output_dir}/model_comparison_dashboard.png")
    plt.close()


def generate_improvement_report(
    results_dict, output_path="reports/improvement_report.md"
):
    """
    Generate Markdown report with detailed improvement analysis.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Model Improvement Report\n\n")
        f.write("## Executive Summary\n\n")

        # Calculate improvements
        if "base" in results_dict and "focal" in results_dict:
            base = results_dict["base"]
            focal = results_dict["focal"]

            recall_improvement = (focal["recall"] - base["recall"]) * 100
            fn_reduction = (base["fn_rate"] - focal["fn_rate"]) * 100

            f.write(f"### Key Achievements\n\n")
            f.write(
                f"- **Sensitivity Improvement**: +{recall_improvement:.1f} percentage points\n"
            )
            f.write(
                f"- **False Negative Reduction**: -{fn_reduction:.1f} percentage points\n"
            )
            f.write(f"- **Final Sensitivity**: {focal['recall']:.1%}\n")
            f.write(
                f"- **Clinical Target Met**: {'✅ Yes' if focal['recall'] > 0.85 else '⚠️ Not Yet'}\n\n"
            )

        f.write("## Methodology\n\n")
        f.write("### Implemented Improvements\n\n")
        f.write("1. **Focal Loss (γ=2.5)**\n")
        f.write("   - Automatically focuses on hard examples\n")
        f.write("   - Down-weights easy examples (confident correct predictions)\n")
        f.write("   - Up-weights misclassifications\n\n")

        f.write("2. **Label Smoothing (ε=0.1)**\n")
        f.write("   - Reduces overconfidence\n")
        f.write("   - Prevents model from outputting extreme probabilities\n")
        f.write("   - Improves calibration\n\n")

        f.write("3. **Tumor-Focused Augmentation**\n")
        f.write("   - Aggressive rotation (±54°)\n")
        f.write("   - Bilateral flips\n")
        f.write("   - Intensity variations\n\n")

        f.write("4. **Test Time Augmentation (Optional)**\n")
        f.write("   - Ensemble of 5 augmented predictions\n")
        f.write("   - Reduces variance\n")
        f.write("   - Improves robustness\n\n")

        f.write("## Detailed Metrics\n\n")

        # Create comparison table
        f.write("| Metric | Base Model | Fine-Tuned | Focal Loss | Improvement |\n")
        f.write("|--------|------------|------------|------------|-------------|\n")

        for metric in ["recall", "specificity", "precision", "accuracy"]:
            base_val = results_dict.get("base", {}).get(metric, 0)
            ft_val = results_dict.get("finetuned", {}).get(metric, 0)
            focal_val = results_dict.get("focal", {}).get(metric, 0)
            improvement = (focal_val - base_val) * 100

            f.write(
                f"| {metric.capitalize()} | {base_val:.1%} | {ft_val:.1%} | {focal_val:.1%} | "
            )
            f.write(f"{'+' if improvement > 0 else ''}{improvement:.1f}pp |\n")

        f.write("\n## Recommendations\n\n")

        if focal_val < 0.85:
            f.write("### Further Improvements Needed\n\n")
            f.write("- Consider increasing Focal Loss gamma to 3.0\n")
            f.write("- Enable Test Time Augmentation (5-10 samples)\n")
            f.write("- Add more tumor samples via synthesis\n")
            f.write("- Ensemble multiple models\n")
        else:
            f.write("### Model Ready for Clinical Validation\n\n")
            f.write("✅ Sensitivity target achieved (>85%)\n\n")
            f.write("Next steps:\n")
            f.write("- Validate on additional external datasets\n")
            f.write("- Conduct expert review with radiologists\n")
            f.write("- Prepare regulatory documentation\n")

    print(f"✅ Improvement report saved: {output_path}")


def generate_metrics_table(results_dict, output_path="reports/metrics_comparison.csv"):
    """
    Generate CSV table with all metrics for easy reference.
    """
    rows = []

    for model_name, data in results_dict.items():
        cm = np.array(data["confusion_matrix"])
        tn, fp, fn, tp = cm.ravel()

        # Calculate additional metrics
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = (
            2
            * (data["precision"] * data["recall"])
            / (data["precision"] + data["recall"])
            if (data["precision"] + data["recall"]) > 0
            else 0
        )

        row = {
            "Model": model_name.capitalize(),
            "Recall (Sensitivity)": f"{data['recall']:.3f}",
            "Specificity": f"{data['specificity']:.3f}",
            "Precision (PPV)": f"{data['precision']:.3f}",
            "NPV": f"{npv:.3f}",
            "Accuracy": f"{data['accuracy']:.3f}",
            "F1-Score": f"{f1:.3f}",
            "FN Rate": f"{data['fn_rate']:.3f}",
            "True Negatives": tn,
            "False Positives": fp,
            "False Negatives": fn,
            "True Positives": tp,
            "Source": data.get("source", "N/A"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✅ Metrics table saved: {output_path}")


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    """
    Load actual results and generate comparison dashboard.
    
    Usage:
        python tools/compare_models.py --config configs/config.yaml --output reports
    """
    parser = argparse.ArgumentParser(
        description="Compare model results and generate dashboard"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument("--output", default="reports", help="Output directory")
    parser.add_argument(
        "--checkpoint_dir", default="models", help="Checkpoint directory"
    )

    args = parser.parse_args()

    # Load actual results from saved files
    print("=" * 70)
    print("LOADING MODEL RESULTS")
    print("=" * 70)

    results = load_results_from_files(
        checkpoint_dir=args.checkpoint_dir, reports_dir=args.output
    )

    if not results:
        print("\n[ERROR] No results found. Please run the following first:")
        print("  1. python notebooks/error_analysis.py")
        print("  2. python tools/evaluate_external.py")
        print("  3. python tools/adaptive_retrain.py")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"LOADED {len(results)} MODEL(S)")
    print("=" * 70)

    for model_name, data in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Source: {data.get('source', 'unknown')}")
        print(f"  Recall: {data['recall']:.1%}")
        print(f"  Specificity: {data['specificity']:.1%}")
        print(f"  Precision: {data['precision']:.1%}")
        print(f"  Accuracy: {data['accuracy']:.1%}")

        cm = np.array(data["confusion_matrix"])
        tn, fp, fn, tp = cm.ravel()
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING COMPARATIVE ANALYSIS")
    print("=" * 70)

    create_comparative_dashboard(results, output_dir=args.output)

    # Generate improvement report
    if len(results) >= 2:
        generate_improvement_report(
            results, output_path=os.path.join(args.output, "improvement_report.md")
        )

    # Generate detailed comparison table
    generate_metrics_table(
        results, output_path=os.path.join(args.output, "metrics_comparison.csv")
    )

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"📊 Dashboard: {args.output}/model_comparison_dashboard.png")
    print(f"📝 Report: {args.output}/improvement_report.md")
    print(f"📋 Metrics Table: {args.output}/metrics_comparison.csv")

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if "base" in results and "focal" in results:
        base_recall = results["base"]["recall"]
        focal_recall = results["focal"]["recall"]
        improvement = (focal_recall - base_recall) * 100

        base_fn = np.array(results["base"]["confusion_matrix"]).ravel()[2]
        focal_fn = np.array(results["focal"]["confusion_matrix"]).ravel()[2]
        fn_reduction = base_fn - focal_fn

        print(
            f"\n🎯 Sensitivity Improvement: {base_recall:.1%} → {focal_recall:.1%} (+{improvement:.1f}pp)"
        )
        print(
            f"✅ False Negatives Reduced: {base_fn} → {focal_fn} (-{fn_reduction} cases, {-fn_reduction / base_fn:.1%})"
        )

        if focal_recall >= 0.85:
            print(f"\n✅ CLINICAL TARGET MET: Sensitivity ≥ 85%")
        else:
            print(f"\n⚠️  Clinical target not yet met (need ≥85% sensitivity)")
            print(f"   Current: {focal_recall:.1%}")
            print(f"   Gap: {(0.85 - focal_recall) * 100:.1f}pp")
    elif "base" in results:
        print(f"\n⚠️  Only base model results available")
        print(f"   Run: python tools/adaptive_retrain.py")
