"""Upload the NOBLE benchmark to HuggingFace Hub.

Steps:
  1. Build a manifest CSV listing every image with full metadata
  2. Copy dataset_card.md to README.md for HF
  3. Upload data/ + configs/prompts/ + manifest.csv + README.md

Output goes to: ashleyscruse/noble-ai-evidence-benchmark (repo_type=dataset)

The upload is resumable. If interrupted, just re-run; it skips files already on the Hub.
"""

import csv
import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "ashleyscruse/noble-ai-evidence-benchmark"
PROJECT_ROOT = Path("/work2/10539/ashleyscruse/vista/ai-generated-image-detection")
DATA_DIR = PROJECT_ROOT / "data"
DATASET_CARD = PROJECT_ROOT / "dataset_card.md"
README_DEST = PROJECT_ROOT / "README.md"
MANIFEST_DEST = PROJECT_ROOT / "manifest.csv"

# Filename pattern for synthetic images: <model>_p<prompt_id>_v<variation_id>.png
SYNTH_PATTERN = re.compile(r"^(?P<model>[a-z0-9]+)_p(?P<prompt>\d+)_v(?P<variation>\d+)")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_synth_filename(name: str) -> dict:
    """Extract model/prompt/variation IDs from synthetic filename."""
    m = SYNTH_PATTERN.match(name.lower())
    if not m:
        return {"generator": None, "prompt_id": None, "variation_id": None}
    return {
        "generator": m.group("model"),
        "prompt_id": int(m.group("prompt")),
        "variation_id": int(m.group("variation")),
    }


def build_manifest() -> int:
    """Scan all images and write manifest.csv. Returns row count."""
    rows = []

    print("Scanning processed/ ...")
    for level in ["clean", "moderate", "heavy"]:
        for source in ["real", "synthetic"]:
            base = DATA_DIR / "processed" / level / source
            if not base.exists():
                continue
            for img in sorted(base.rglob("*")):
                if img.suffix.lower() not in IMAGE_EXT:
                    continue
                rel_to_source = img.relative_to(base)
                category = rel_to_source.parts[0] if len(rel_to_source.parts) > 1 else "uncategorized"

                row = {
                    "path": str(img.relative_to(PROJECT_ROOT)),
                    "label": source,
                    "split": "processed",
                    "level": level,
                    "category": category,
                    "generator": None,
                    "prompt_id": None,
                    "variation_id": None,
                }
                if source == "synthetic":
                    row.update(parse_synth_filename(img.name))
                rows.append(row)

    print("Scanning raw/real/ ...")
    raw_real = DATA_DIR / "raw" / "real"
    if raw_real.exists():
        for img in sorted(raw_real.rglob("*")):
            if img.suffix.lower() not in IMAGE_EXT:
                continue
            rel = img.relative_to(raw_real)
            category = rel.parts[1] if len(rel.parts) > 1 else rel.parts[0] if rel.parts else "uncategorized"
            rows.append({
                "path": str(img.relative_to(PROJECT_ROOT)),
                "label": "real",
                "split": "raw",
                "level": "none",
                "category": category,
                "generator": None,
                "prompt_id": None,
                "variation_id": None,
            })

    print("Scanning raw/synthetic/ ...")
    raw_synth = DATA_DIR / "raw" / "synthetic"
    if raw_synth.exists():
        for img in sorted(raw_synth.rglob("*")):
            if img.suffix.lower() not in IMAGE_EXT:
                continue
            rel = img.relative_to(raw_synth)
            category = rel.parts[0] if rel.parts else "uncategorized"
            row = {
                "path": str(img.relative_to(PROJECT_ROOT)),
                "label": "synthetic",
                "split": "raw",
                "level": "none",
                "category": category,
                "generator": None,
                "prompt_id": None,
                "variation_id": None,
            }
            row.update(parse_synth_filename(img.name))
            rows.append(row)

    fieldnames = ["path", "label", "split", "level", "category", "generator", "prompt_id", "variation_id"]
    with open(MANIFEST_DEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest written: {MANIFEST_DEST} ({len(rows):,} rows)")
    return len(rows)


def copy_dataset_card_as_readme():
    """HF datasets expect README.md, not dataset_card.md."""
    if not DATASET_CARD.exists():
        print(f"WARNING: {DATASET_CARD} not found, skipping README copy")
        return
    shutil.copy(DATASET_CARD, README_DEST)
    print(f"Copied {DATASET_CARD.name} to README.md for HF")


def upload():
    """Upload everything to HF Hub."""
    api = HfApi()

    print(f"\nUploading to {REPO_ID} ...")
    print("(This is resumable. If interrupted, re-run the script.)")
    print()

    # Create the repo if it doesn't exist (no-op if it does)
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    # Upload everything except things we want to skip
    api.upload_folder(
        folder_path=str(PROJECT_ROOT),
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=[
            "README.md",
            "manifest.csv",
            "data/processed/**",
            "data/raw/real/**",
            "data/raw/synthetic/**",
            "configs/prompts/**",
        ],
        ignore_patterns=[
            "data/raw/videos/**",
            "data/raw/synthetic/generation_log_*.json",
            "**/.DS_Store",
            "**/.gitkeep",
            "**/__pycache__/**",
        ],
        commit_message="v1.1 release: 3 generators, 3 degradation levels, 74,184 instances",
    )

    print()
    print(f"Upload complete: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    n_rows = build_manifest()
    copy_dataset_card_as_readme()
    print(f"\nReady to upload {n_rows:,} images plus prompts + README + manifest.")
    upload()
