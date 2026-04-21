"""Generate a comprehensive funder-ready report with detailed figures.

Creates:
  - Per-generator performance breakdown
  - Per-category performance breakdown
  - Accuracy degradation curves
  - Confusion matrices per level
  - Summary statistics table

Usage:
    python -m src.evaluation.generate_report
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.config import get_results_path

LEVELS = ["clean", "moderate", "heavy"]
LEVEL_LABELS = {"clean": "Clean\n(Original)", "moderate": "Moderate\n(JPEG+Blur)", "heavy": "Heavy\n(Compression+Noise)"}


def load_all_results():
    """Load all result JSON files into a combined DataFrame."""
    metrics_dir = get_results_path("metrics")
    all_rows = []

    for level in LEVELS:
        result_file = metrics_dir / f"results_huggingface_{level}.json"
        if not result_file.exists():
            print(f"  Missing: {result_file}")
            continue

        with open(result_file) as f:
            data = json.load(f)

        for pred in data.get("predictions", []):
            if pred["prediction"] == "error":
                continue

            path = pred["path"]

            # Determine ground truth from path
            if "/real/" in path:
                label = "real"
            elif "/synthetic/" in path:
                label = "synthetic"
            else:
                continue

            # Determine generator from filename
            fname = Path(path).stem
            if fname.startswith("sd15_"):
                generator = "SD 1.5"
            elif fname.startswith("openjourney_"):
                generator = "OpenJourney"
            elif fname.startswith("realistic_"):
                generator = "Realistic Vision"
            else:
                generator = "N/A"

            # Determine category from path
            parts = Path(path).parts
            try:
                if "real" in parts:
                    idx = parts.index("real")
                elif "synthetic" in parts:
                    idx = parts.index("synthetic")
                else:
                    idx = -2
                category = parts[idx + 1] if idx + 1 < len(parts) - 1 else "unknown"
            except (ValueError, IndexError):
                category = "unknown"

            all_rows.append({
                "path": path,
                "level": level,
                "label": label,
                "prediction": pred["prediction"],
                "confidence": pred["confidence"],
                "generator": generator,
                "category": category,
            })

    return pd.DataFrame(all_rows)


def plot_accuracy_by_level(df, figures_dir):
    """Bar chart: overall accuracy at each degradation level."""
    fig, ax = plt.subplots(figsize=(8, 5))

    accuracies = []
    for level in LEVELS:
        level_df = df[df["level"] == level]
        y_true = (level_df["label"] == "synthetic").astype(int)
        y_pred = (level_df["prediction"] == "synthetic").astype(int)
        accuracies.append(accuracy_score(y_true, y_pred))

    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax.bar([LEVEL_LABELS[l] for l in LEVELS], accuracies, color=colors, width=0.5)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Random chance")
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Detection Accuracy Degrades with Image Quality", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f"{acc:.1%}", ha="center", fontsize=12, fontweight="bold")

    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(figures_dir / "accuracy_degradation.png", dpi=200)
    plt.close()
    print("  Saved: accuracy_degradation.png")


def plot_per_generator(df, figures_dir):
    """Grouped bar chart: accuracy by generator and degradation level."""
    fig, ax = plt.subplots(figsize=(10, 6))

    generators = ["SD 1.5", "OpenJourney", "Realistic Vision"]
    x = np.arange(len(generators))
    width = 0.25

    for i, level in enumerate(LEVELS):
        level_df = df[df["level"] == level]
        accs = []
        for gen in generators:
            gen_df = level_df[(level_df["generator"] == gen) | (level_df["label"] == "real")]
            y_true = (gen_df["label"] == "synthetic").astype(int)
            y_pred = (gen_df["prediction"] == "synthetic").astype(int)
            accs.append(accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0)

        colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        bars = ax.bar(x + i * width, accs, width, label=level.capitalize(), color=colors[i])

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Detection Accuracy by Generator and Quality Level", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(generators, fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(title="Quality Level", fontsize=11)
    plt.tight_layout()
    fig.savefig(figures_dir / "accuracy_by_generator.png", dpi=200)
    plt.close()
    print("  Saved: accuracy_by_generator.png")


def plot_confusion_matrices(df, figures_dir):
    """Confusion matrices for each degradation level."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, level in zip(axes, LEVELS):
        level_df = df[df["level"] == level]
        y_true = (level_df["label"] == "synthetic").astype(int)
        y_pred = (level_df["prediction"] == "synthetic").astype(int)

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Real", "Synthetic"], yticklabels=["Real", "Synthetic"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{level.capitalize()}", fontsize=13, fontweight="bold")

    fig.suptitle("Confusion Matrices by Degradation Level", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / "confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: confusion_matrices.png")


def plot_roc_curves(df, figures_dir):
    """ROC curves per degradation level."""
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    for level, color in zip(LEVELS, colors):
        level_df = df[df["level"] == level]
        y_true = (level_df["label"] == "synthetic").astype(int)
        y_scores = level_df["confidence"].values

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{level.capitalize()} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves by Degradation Level", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    fig.savefig(figures_dir / "roc_curves.png", dpi=200)
    plt.close()
    print("  Saved: roc_curves.png")


def plot_metrics_summary(df, figures_dir):
    """Summary table as a figure."""
    rows = []
    for level in LEVELS:
        level_df = df[df["level"] == level]
        y_true = (level_df["label"] == "synthetic").astype(int)
        y_pred = (level_df["prediction"] == "synthetic").astype(int)
        y_scores = level_df["confidence"].values

        row = {
            "Quality Level": level.capitalize(),
            "N": len(level_df),
            "Accuracy": f"{accuracy_score(y_true, y_pred):.1%}",
            "Precision": f"{precision_score(y_true, y_pred, zero_division=0):.1%}",
            "Recall": f"{recall_score(y_true, y_pred, zero_division=0):.1%}",
            "F1": f"{f1_score(y_true, y_pred, zero_division=0):.1%}",
        }

        if len(np.unique(y_true)) == 2:
            row["AUC-ROC"] = f"{roc_auc_score(y_true, y_scores):.3f}"
        else:
            row["AUC-ROC"] = "N/A"

        rows.append(row)

    table_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(table_df.columns)):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = "#ecf0f1" if i % 2 == 0 else "white"
        for j in range(len(table_df.columns)):
            table[i, j].set_facecolor(color)

    ax.set_title("HuggingFace AI Detector Performance on Law Enforcement Benchmark",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(figures_dir / "metrics_table.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: metrics_table.png")


def plot_dataset_overview(df, figures_dir):
    """Dataset composition visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Real image categories
    clean_df = df[df["level"] == "clean"]
    real_cats = clean_df[clean_df["label"] == "real"]["category"].value_counts()
    ax1.barh(real_cats.index, real_cats.values, color="#3498db")
    ax1.set_xlabel("Count")
    ax1.set_title("Real Images by Category", fontweight="bold")
    for i, v in enumerate(real_cats.values):
        ax1.text(v + 1, i, str(v), va="center")

    # Synthetic by generator
    synth_df = clean_df[clean_df["label"] == "synthetic"]
    gen_counts = synth_df["generator"].value_counts()
    colors = ["#e74c3c", "#f39c12", "#9b59b6"]
    ax2.barh(gen_counts.index, gen_counts.values, color=colors[:len(gen_counts)])
    ax2.set_xlabel("Count")
    ax2.set_title("Synthetic Images by Generator", fontweight="bold")
    for i, v in enumerate(gen_counts.values):
        ax2.text(v + 1, i, str(v), va="center")

    fig.suptitle(f"Benchmark Dataset: {len(clean_df)} Images", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(figures_dir / "dataset_overview.png", dpi=200)
    plt.close()
    print("  Saved: dataset_overview.png")


def main():
    print("Loading results...")
    df = load_all_results()

    if df.empty:
        print("No results found. Run evaluation first.")
        return

    print(f"Loaded {len(df)} predictions across {df['level'].nunique()} levels")

    figures_dir = get_results_path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating funder report figures...")
    plot_dataset_overview(df, figures_dir)
    plot_accuracy_by_level(df, figures_dir)
    plot_per_generator(df, figures_dir)
    plot_confusion_matrices(df, figures_dir)
    plot_roc_curves(df, figures_dir)
    plot_metrics_summary(df, figures_dir)

    # Save summary CSV
    summary_path = get_results_path("metrics") / "funder_summary.csv"
    summary_rows = []
    for level in LEVELS:
        level_df = df[df["level"] == level]
        y_true = (level_df["label"] == "synthetic").astype(int)
        y_pred = (level_df["prediction"] == "synthetic").astype(int)
        y_scores = level_df["confidence"].values

        summary_rows.append({
            "level": level,
            "n_images": len(level_df),
            "n_real": (level_df["label"] == "real").sum(),
            "n_synthetic": (level_df["label"] == "synthetic").sum(),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) == 2 else None,
        })

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\n  Saved: {summary_path}")
    print(f"\nAll figures saved to: {figures_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
