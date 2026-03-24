"""Download and filter real images from COCO dataset for law enforcement categories.

COCO is a good complement to Open Images — it has more scene-level images
(rooms, streets, parking lots) which are useful for our indoor/outdoor categories.

Usage:
    # Download all categories
    python -m src.data_collection.download_coco

    # Download a specific category
    python -m src.data_collection.download_coco --category indoor_scenes --max-images 100

    # List available COCO classes for a category
    python -m src.data_collection.download_coco --list-classes

    # Dry run
    python -m src.data_collection.download_coco --category vehicles --max-images 50 --dry-run

Prerequisites:
    pip install pycocotools
    (already included in requirements.txt)
"""

import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import CONFIG, get_data_path, get_project_root

# Mapping from our project categories to COCO category names.
# Full list of COCO categories: https://cocodataset.org/#explore
CATEGORY_TO_CLASSES = {
    "people": [
        "person",
    ],
    "vehicles": [
        "car", "bus", "truck", "motorcycle", "bicycle",
    ],
    "indoor_scenes": [
        "couch", "bed", "dining table", "toilet", "tv",
        "microwave", "oven", "refrigerator", "sink",
    ],
    "outdoor_scenes": [
        "traffic light", "stop sign", "parking meter",
        "fire hydrant", "bench",
    ],
    "objects": [
        "backpack", "handbag", "suitcase", "bottle",
        "knife", "cell phone", "laptop", "book",
    ],
}

COCO_ANNOTATIONS_URL = {
    "train": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
COCO_IMAGE_BASE_URL = "http://images.cocodataset.org/train2017"


def get_cache_dir() -> Path:
    """Return the cache directory for downloaded metadata files."""
    cache_dir = get_project_root() / ".cache" / "coco"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_coco_annotations() -> dict:
    """Download and cache COCO annotations.

    Returns the parsed JSON for the train2017 instances annotations.
    """
    cache_dir = get_cache_dir()
    annotations_path = cache_dir / "instances_train2017.json"

    if not annotations_path.exists():
        zip_path = cache_dir / "annotations_trainval2017.zip"

        if not zip_path.exists():
            print("Downloading COCO annotations (~252 MB, be patient)...")
            urllib.request.urlretrieve(COCO_ANNOTATIONS_URL["train"], zip_path)

        print("Extracting annotations...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract just the file we need
            for member in zf.namelist():
                if "instances_train2017.json" in member:
                    with zf.open(member) as src, open(annotations_path, "wb") as dst:
                        dst.write(src.read())
                    break

    print("Loading COCO annotations...")
    with open(annotations_path) as f:
        return json.load(f)


def list_all_classes() -> None:
    """Print all COCO categories and which project category they map to."""
    coco = load_coco_annotations()

    # Build reverse mapping
    class_to_category = {}
    for cat, classes in CATEGORY_TO_CLASSES.items():
        for cls in classes:
            class_to_category[cls] = cat

    print(f"\n{'COCO Class':<25} {'Project Category':<20} {'COCO ID'}")
    print("-" * 60)
    for cat_info in sorted(coco["categories"], key=lambda x: x["name"]):
        name = cat_info["name"]
        project_cat = class_to_category.get(name, "—")
        print(f"{name:<25} {project_cat:<20} {cat_info['id']}")


def find_images_for_category(
    coco: dict,
    category: str,
    max_images: int = None,
    seed: int = None,
) -> pd.DataFrame:
    """Find COCO image IDs containing objects from a given category.

    Args:
        coco: Parsed COCO annotations dict
        category: One of the keys in CATEGORY_TO_CLASSES
        max_images: Maximum number of unique images to return
        seed: Random seed for reproducible sampling

    Returns:
        DataFrame with columns: image_id, file_name, coco_url, classes_present
    """
    if seed is None:
        seed = CONFIG.get("random_seed", 42)

    target_class_names = CATEGORY_TO_CLASSES[category]

    # Build category name -> id mapping
    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    target_cat_ids = [cat_name_to_id[n] for n in target_class_names if n in cat_name_to_id]

    if not target_cat_ids:
        print(f"No matching COCO classes for category: {category}")
        return pd.DataFrame()

    # Find image IDs with these categories
    image_to_classes = {}
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    for ann in coco["annotations"]:
        if ann["category_id"] in target_cat_ids:
            img_id = ann["image_id"]
            class_name = cat_id_to_name[ann["category_id"]]
            if img_id not in image_to_classes:
                image_to_classes[img_id] = set()
            image_to_classes[img_id].add(class_name)

    image_ids = list(image_to_classes.keys())
    print(f"Found {len(image_ids)} images with {category} objects")

    # Sample if needed
    if max_images and len(image_ids) > max_images:
        rng = np.random.default_rng(seed)
        image_ids = rng.choice(image_ids, size=max_images, replace=False).tolist()
        print(f"Sampled {max_images} images")

    # Build image info lookup
    image_info = {img["id"]: img for img in coco["images"]}

    rows = []
    for img_id in image_ids:
        info = image_info.get(img_id)
        if info:
            rows.append({
                "image_id": img_id,
                "file_name": info["file_name"],
                "coco_url": info["coco_url"],
                "classes_present": ", ".join(sorted(image_to_classes.get(img_id, set()))),
            })

    return pd.DataFrame(rows)


def download_images(
    image_df: pd.DataFrame,
    output_dir: Path,
    skip_existing: bool = True,
) -> dict:
    """Download COCO images.

    Args:
        image_df: DataFrame with file_name and coco_url columns
        output_dir: Directory to save images
        skip_existing: Skip images that already exist on disk

    Returns:
        Dict with download stats
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    for _, row in tqdm(image_df.iterrows(), total=len(image_df), desc="Downloading"):
        file_name = row["file_name"]
        url = row["coco_url"]
        output_path = output_dir / file_name

        if skip_existing and output_path.exists():
            stats["skipped"] += 1
            continue

        try:
            urllib.request.urlretrieve(url, output_path)
            stats["downloaded"] += 1
        except Exception as e:
            stats["failed"] += 1
            if stats["failed"] <= 10:
                print(f"  Failed: {file_name} - {e}")
            elif stats["failed"] == 11:
                print("  (suppressing further error messages)")

    return stats


def save_manifest(image_df: pd.DataFrame, output_dir: Path, category: str) -> None:
    """Save a CSV manifest of downloaded images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"manifest_coco_{category}.csv"
    image_df.to_csv(manifest_path, index=False)
    print(f"Saved manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download real images from COCO for the NOBLE benchmark."
    )
    parser.add_argument(
        "--category",
        choices=list(CATEGORY_TO_CLASSES.keys()),
        help="Image category to download. If not specified, downloads all.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Max images to download. Defaults to config.yaml target.",
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="List all COCO categories and their project mapping.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find images but don't download. Saves manifest only.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed (default: {CONFIG.get('random_seed', 42)} from config).",
    )

    args = parser.parse_args()

    # Load annotations once (shared across categories)
    if args.list_classes:
        list_all_classes()
        return

    coco = load_coco_annotations()

    # Determine categories
    if args.category:
        categories = [args.category]
    else:
        categories = list(CATEGORY_TO_CLASSES.keys())

    output_base = get_data_path("raw/real")
    real_targets = CONFIG["dataset"]["real_categories"]

    for category in categories:
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")

        max_images = args.max_images or real_targets.get(category, 1000)

        image_df = find_images_for_category(
            coco=coco,
            category=category,
            max_images=max_images,
            seed=args.seed,
        )

        if image_df.empty:
            print(f"No images found for {category}, skipping.")
            continue

        print(f"Found {len(image_df)} images to download")

        category_dir = output_base / category
        save_manifest(image_df, category_dir, category)

        if args.dry_run:
            print("Dry run — skipping download.")
            continue

        stats = download_images(image_df, category_dir)
        print(f"Results: {stats['downloaded']} downloaded, {stats['skipped']} skipped, {stats['failed']} failed")

    print("\nDone!")


if __name__ == "__main__":
    main()
