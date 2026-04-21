"""Evaluate AI-generated image detection tools on the benchmark dataset.

Supports multiple detection backends:
  - HuggingFace model (umm-maybe/AI-image-detector) -- runs locally
  - Hive Moderation API
  - AI or Not API

Usage:
    # Evaluate HuggingFace detector on all dataset versions
    python -m src.evaluation.evaluate --tool huggingface

    # Evaluate on a specific degradation level
    python -m src.evaluation.evaluate --tool huggingface --level clean

    # Evaluate Hive API (requires HIVE_API_KEY in .env)
    python -m src.evaluation.evaluate --tool hive --level clean

    # Evaluate AI or Not API
    python -m src.evaluation.evaluate --tool aiornot --level clean

    # Generate figures and summary report from saved results
    python -m src.evaluation.evaluate --report

    # Dry run: show dataset stats without running detection
    python -m src.evaluation.evaluate --tool huggingface --dry-run
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import CONFIG, get_data_path, get_results_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

LEVELS = ["clean", "moderate", "heavy"]

TOOLS = ["huggingface", "hive", "aiornot"]


def find_images(directory: Path) -> list[Path]:
    """Find all image files recursively."""
    if not directory.exists():
        return []
    return sorted(
        f for f in directory.rglob("*")
        if f.suffix.lower() in IMAGE_EXTENSIONS
        and not f.name.startswith(".")
        and "_duplicates" not in str(f)
    )


def build_dataset_manifest(level: str) -> pd.DataFrame:
    """Build a manifest of all images at a given degradation level.

    Returns a DataFrame with columns: path, label (real/synthetic), category, level.
    """
    rows = []
    base = get_data_path(f"processed/{level}")

    for source in ["real", "synthetic"]:
        source_dir = base / source
        for img_path in find_images(source_dir):
            # Extract category from subdirectory
            relative = img_path.relative_to(source_dir)
            category = relative.parts[0] if len(relative.parts) > 1 else "uncategorized"

            rows.append({
                "path": str(img_path),
                "label": source,  # ground truth: "real" or "synthetic"
                "category": category,
                "level": level,
            })

    return pd.DataFrame(rows)


def detect_huggingface(image_paths: list[str], batch_size: int = 16) -> list[dict]:
    """Run the HuggingFace AI image detector locally.

    Uses the umm-maybe/AI-image-detector model (ViT-based classifier).

    Returns:
        List of dicts with keys: path, prediction, confidence, raw_scores.
    """
    from PIL import Image
    from transformers import pipeline

    print("Loading HuggingFace detector: umm-maybe/AI-image-detector")
    detector = pipeline(
        "image-classification",
        model="umm-maybe/AI-image-detector",
        device=0 if _has_gpu() else -1,
    )

    results = []
    from tqdm import tqdm

    for i in tqdm(range(0, len(image_paths), batch_size), desc="HuggingFace detector"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_images.append(img)
            except Exception as e:
                results.append({
                    "path": p,
                    "prediction": "error",
                    "confidence": 0.0,
                    "raw_scores": {"error": str(e)},
                })
                continue

        if not batch_images:
            continue

        try:
            outputs = detector(batch_images, top_k=2)

            for path, output in zip(batch_paths[:len(outputs)], outputs):
                # output is a list of {label, score} dicts
                scores = {item["label"].lower(): item["score"] for item in output}

                # The model outputs "artificial" and "human" labels
                ai_score = scores.get("artificial", 0.0)
                human_score = scores.get("human", 0.0)

                prediction = "synthetic" if ai_score >= 0.5 else "real"
                confidence = ai_score  # always store P(synthetic)

                results.append({
                    "path": path,
                    "prediction": prediction,
                    "confidence": confidence,
                    "raw_scores": scores,
                })

        except Exception as e:
            for p in batch_paths:
                results.append({
                    "path": p,
                    "prediction": "error",
                    "confidence": 0.0,
                    "raw_scores": {"error": str(e)},
                })

    return results


def detect_hive(image_paths: list[str]) -> list[dict]:
    """Run Hive Moderation API for AI detection.

    Requires HIVE_API_KEY in .env.
    """
    import requests
    from dotenv import load_dotenv

    load_dotenv(get_data_path("..") / ".env")

    api_key = os.environ.get("HIVE_API_KEY")
    if not api_key:
        raise EnvironmentError("HIVE_API_KEY not found in .env or environment.")

    results = []
    from tqdm import tqdm

    for path in tqdm(image_paths, desc="Hive API"):
        try:
            with open(path, "rb") as f:
                response = requests.post(
                    "https://api.hivemoderation.com/api/v1/task/sync",
                    headers={"Authorization": f"Token {api_key}"},
                    files={"media": f},
                )

            if response.status_code == 200:
                data = response.json()
                # Navigate Hive response structure
                ai_score = _extract_hive_ai_score(data)
                prediction = "synthetic" if ai_score > 0.5 else "real"

                results.append({
                    "path": path,
                    "prediction": prediction,
                    "confidence": ai_score if prediction == "synthetic" else 1 - ai_score,
                    "raw_scores": {"ai_generated": ai_score},
                })
            else:
                results.append({
                    "path": path,
                    "prediction": "error",
                    "confidence": 0.0,
                    "raw_scores": {"error": f"HTTP {response.status_code}"},
                })

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            results.append({
                "path": path,
                "prediction": "error",
                "confidence": 0.0,
                "raw_scores": {"error": str(e)},
            })

    return results


def detect_aiornot(image_paths: list[str]) -> list[dict]:
    """Run AI or Not API for AI detection.

    Requires AIORNOT_API_KEY in .env.
    """
    import requests
    from dotenv import load_dotenv

    load_dotenv(get_data_path("..") / ".env")

    api_key = os.environ.get("AIORNOT_API_KEY")
    if not api_key:
        raise EnvironmentError("AIORNOT_API_KEY not found in .env or environment.")

    results = []
    from tqdm import tqdm

    for path in tqdm(image_paths, desc="AI or Not API"):
        try:
            with open(path, "rb") as f:
                response = requests.post(
                    "https://api.aiornot.com/v1/reports/image",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"object": f},
                )

            if response.status_code == 200:
                data = response.json()
                verdict = data.get("report", {}).get("verdict", "unknown")
                ai_score = data.get("report", {}).get("ai", {}).get("confidence", 0.5)

                prediction = "synthetic" if verdict == "ai" else "real"
                results.append({
                    "path": path,
                    "prediction": prediction,
                    "confidence": ai_score,
                    "raw_scores": data.get("report", {}),
                })
            else:
                results.append({
                    "path": path,
                    "prediction": "error",
                    "confidence": 0.0,
                    "raw_scores": {"error": f"HTTP {response.status_code}"},
                })

            time.sleep(1)

        except Exception as e:
            results.append({
                "path": path,
                "prediction": "error",
                "confidence": 0.0,
                "raw_scores": {"error": str(e)},
            })

    return results


def compute_metrics(manifest: pd.DataFrame, results: list[dict]) -> dict:
    """Compute evaluation metrics from detection results.

    Returns dict with overall and per-category metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    # Merge results with manifest
    results_df = pd.DataFrame(results)
    results_df = results_df[results_df["prediction"] != "error"]

    if results_df.empty:
        return {"error": "No valid predictions"}

    merged = manifest.merge(results_df, on="path", how="inner")

    # Binary labels: synthetic=1, real=0
    y_true = (merged["label"] == "synthetic").astype(int)
    y_pred = (merged["prediction"] == "synthetic").astype(int)
    y_scores = merged["confidence"].values

    # Confidence is already P(synthetic) from all detectors
    y_scores_synthetic = y_scores

    metrics = {
        "overall": {
            "n_samples": len(merged),
            "n_real": int((y_true == 0).sum()),
            "n_synthetic": int((y_true == 1).sum()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        },
    }

    # AUC-ROC (requires both classes present)
    if len(np.unique(y_true)) == 2:
        metrics["overall"]["auc_roc"] = float(roc_auc_score(y_true, y_scores_synthetic))

    # Per-category breakdown
    metrics["by_category"] = {}
    for cat in merged["category"].unique():
        mask = merged["category"] == cat
        if mask.sum() < 2:
            continue

        cat_true = y_true[mask]
        cat_pred = y_pred[mask]

        cat_metrics = {
            "n_samples": int(mask.sum()),
            "accuracy": float(accuracy_score(cat_true, cat_pred)),
            "f1": float(f1_score(cat_true, cat_pred, zero_division=0)),
        }

        if len(np.unique(cat_true)) == 2:
            cat_scores = y_scores_synthetic[mask]
            cat_metrics["auc_roc"] = float(roc_auc_score(cat_true, cat_scores))

        metrics["by_category"][cat] = cat_metrics

    # Classification report as string
    metrics["classification_report"] = classification_report(
        y_true, y_pred,
        target_names=["real", "synthetic"],
        zero_division=0,
    )

    return metrics


def generate_figures(results_dir: Path) -> None:
    """Generate visualization figures from saved results."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    figures_dir = get_results_path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Find all result files
    result_files = sorted(results_dir.glob("results_*.json"))
    if not result_files:
        print("No result files found. Run evaluation first.")
        return

    # Collect metrics across tools and levels
    summary_rows = []
    for rf in result_files:
        with open(rf) as f:
            data = json.load(f)
        metrics = data.get("metrics", {}).get("overall", {})
        summary_rows.append({
            "tool": data.get("tool", "unknown"),
            "level": data.get("level", "unknown"),
            "accuracy": metrics.get("accuracy", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1": metrics.get("f1", 0),
            "auc_roc": metrics.get("auc_roc", 0),
        })

    if not summary_rows:
        print("No metrics found in result files.")
        return

    summary_df = pd.DataFrame(summary_rows)

    # 1. Accuracy by tool and degradation level
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = summary_df.pivot(index="tool", columns="level", values="accuracy")
    # Reorder columns
    col_order = [c for c in LEVELS if c in pivot.columns]
    if col_order:
        pivot = pivot[col_order]
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Accuracy")
    ax.set_title("Detection Accuracy by Tool and Degradation Level")
    ax.set_ylim(0, 1)
    ax.legend(title="Degradation Level")
    plt.tight_layout()
    fig.savefig(figures_dir / "accuracy_by_tool_level.png", dpi=150)
    plt.close()
    print(f"  Saved: accuracy_by_tool_level.png")

    # 2. F1 score comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_f1 = summary_df.pivot(index="tool", columns="level", values="f1")
    if col_order:
        pivot_f1 = pivot_f1[[c for c in col_order if c in pivot_f1.columns]]
    pivot_f1.plot(kind="bar", ax=ax)
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score by Tool and Degradation Level")
    ax.set_ylim(0, 1)
    ax.legend(title="Degradation Level")
    plt.tight_layout()
    fig.savefig(figures_dir / "f1_by_tool_level.png", dpi=150)
    plt.close()
    print(f"  Saved: f1_by_tool_level.png")

    # 3. Metric comparison heatmap (one per level)
    for level in col_order:
        level_data = summary_df[summary_df["level"] == level]
        if level_data.empty:
            continue

        metric_cols = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        heatmap_data = level_data.set_index("tool")[metric_cols]

        fig, ax = plt.subplots(figsize=(8, max(3, len(heatmap_data) * 0.8)))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd",
                     vmin=0, vmax=1, ax=ax)
        ax.set_title(f"Detection Metrics -- {level.capitalize()} Images")
        plt.tight_layout()
        fig.savefig(figures_dir / f"metrics_heatmap_{level}.png", dpi=150)
        plt.close()
        print(f"  Saved: metrics_heatmap_{level}.png")

    # 4. Summary table as CSV
    summary_df.to_csv(get_results_path("metrics") / "summary_all_tools.csv", index=False)
    print(f"  Saved: summary_all_tools.csv")

    print(f"\nAll figures saved to {figures_dir}")


def _has_gpu() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _extract_hive_ai_score(response_data: dict) -> float:
    """Extract AI-generated score from Hive API response."""
    try:
        for output in response_data.get("status", []):
            for result in output.get("response", {}).get("output", []):
                for cls in result.get("classes", []):
                    if cls.get("class") == "ai_generated":
                        return cls.get("score", 0.5)
    except (KeyError, TypeError):
        pass
    return 0.5


DETECT_FNS = {
    "huggingface": detect_huggingface,
    "hive": detect_hive,
    "aiornot": detect_aiornot,
}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate detection tools on the NOBLE benchmark."
    )
    parser.add_argument(
        "--tool",
        choices=TOOLS,
        help="Detection tool to evaluate.",
    )
    parser.add_argument(
        "--level",
        choices=LEVELS + ["all"],
        default="all",
        help="Degradation level to evaluate on (default: all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for local models (default: 16).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate figures and summary from saved results (no evaluation).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show dataset stats without running detection.",
    )

    args = parser.parse_args()

    if args.report:
        print("Generating report and figures...")
        generate_figures(get_results_path("metrics"))
        return

    if not args.tool:
        parser.error("Specify --tool or --report")

    levels = LEVELS if args.level == "all" else [args.level]
    detect_fn = DETECT_FNS[args.tool]

    metrics_dir = get_results_path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for level in levels:
        print(f"\n{'='*60}")
        print(f"Tool: {args.tool} | Level: {level}")
        print(f"{'='*60}")

        manifest = build_dataset_manifest(level)

        if manifest.empty:
            print(f"No images found for level: {level}")
            print(f"  Expected at: {get_data_path(f'processed/{level}/')}")
            print(f"  Run degradation pipeline first: python -m src.augmentation.degrade")
            continue

        print(f"Dataset: {len(manifest)} images "
              f"({(manifest['label']=='real').sum()} real, "
              f"{(manifest['label']=='synthetic').sum()} synthetic)")

        if args.dry_run:
            print("\nCategory breakdown:")
            for cat, count in manifest["category"].value_counts().items():
                print(f"  {cat}: {count}")
            continue

        # Run detection
        print(f"\nRunning {args.tool} detector...")
        results = detect_fn(manifest["path"].tolist())

        # Compute metrics
        metrics = compute_metrics(manifest, results)

        # Print summary
        overall = metrics.get("overall", {})
        print(f"\n--- Results ---")
        print(f"Accuracy:  {overall.get('accuracy', 0):.4f}")
        print(f"Precision: {overall.get('precision', 0):.4f}")
        print(f"Recall:    {overall.get('recall', 0):.4f}")
        print(f"F1 Score:  {overall.get('f1', 0):.4f}")
        if "auc_roc" in overall:
            print(f"AUC-ROC:   {overall['auc_roc']:.4f}")
        print(f"\n{metrics.get('classification_report', '')}")

        # Save results
        output_file = metrics_dir / f"results_{args.tool}_{level}.json"
        output_data = {
            "tool": args.tool,
            "level": level,
            "n_images": len(manifest),
            "metrics": metrics,
            "predictions": results,
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"Saved: {output_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
