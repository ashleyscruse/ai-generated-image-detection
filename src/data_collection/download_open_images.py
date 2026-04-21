"""Download and filter real images from Open Images V7 for law enforcement categories.

Usage:
    # Download all categories (uses config.yaml targets)
    python -m src.data_collection.download_open_images

    # Download a specific category
    python -m src.data_collection.download_open_images --category people --max-images 100

    # List available Open Images classes for a category
    python -m src.data_collection.download_open_images --category vehicles --list-classes

    # Dry run (show what would be downloaded, don't download)
    python -m src.data_collection.download_open_images --category people --max-images 50 --dry-run
"""

import argparse
import os
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.config import CONFIG, get_data_path, get_project_root

# Mapping from our project categories to Open Images class names.
# These are case-sensitive and must match the official Open Images V7 class list.
# Full list: https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv
CATEGORY_TO_CLASSES = {
    "people": [
        "Person", "Man", "Woman", "Boy", "Girl",
        "Human face", "Human body",
    ],
    "vehicles": [
        "Car", "Bus", "Truck", "Van", "Ambulance",
        "Motorcycle", "Taxi", "Vehicle registration plate",
    ],
    "indoor_scenes": [
        "Door", "Window", "Stairs", "Table", "Chair",
        "Couch", "Shelf", "Desk", "Bed",
    ],
    "outdoor_scenes": [
        "Building", "House", "Street light", "Traffic sign",
        "Traffic light", "Fence", "Fire hydrant", "Bench",
        "Parking meter", "Tree",
    ],
    "objects": [
        "Knife", "Handgun", "Bag", "Backpack", "Suitcase",
        "Bottle", "Mobile phone", "Laptop", "Camera",
    ],
}

# URLs for Open Images V7 metadata files
CLASS_DESCRIPTIONS_URL = (
    "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
)
BBOX_ANNOTATIONS_URL = {
    "train": "https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-bbox.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
}
IMAGE_IDS_URL = {
    "train": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    "validation": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
}


def get_cache_dir() -> Path:
    """Return the cache directory for downloaded metadata files."""
    cache_dir = get_project_root() / ".cache" / "open_images"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_class_descriptions() -> pd.DataFrame:
    """Download and cache the Open Images class descriptions."""
    cache_path = get_cache_dir() / "class-descriptions-boxable.csv"

    if not cache_path.exists():
        print("Downloading class descriptions...")
        urllib.request.urlretrieve(CLASS_DESCRIPTIONS_URL, cache_path)

    return pd.read_csv(cache_path, header=None, names=["mid", "class_name"])


def load_image_ids(split: str = "train") -> pd.DataFrame:
    """Download and cache the image IDs with URLs."""
    cache_path = get_cache_dir() / f"{split}-images.csv"

    if not cache_path.exists():
        print(f"Downloading {split} image IDs (this may take a minute)...")
        urllib.request.urlretrieve(IMAGE_IDS_URL[split], cache_path)

    return pd.read_csv(cache_path)


def load_annotations(split: str = "train") -> pd.DataFrame:
    """Download and cache bounding box annotations.

    WARNING: The training annotations file is ~2.5 GB. This will take a while
    on the first run. Consider using the validation split for testing.
    """
    cache_path = get_cache_dir() / f"{split}-annotations-bbox.csv"

    if not cache_path.exists():
        print(f"Downloading {split} annotations (training set is ~2.5 GB, be patient)...")
        urllib.request.urlretrieve(BBOX_ANNOTATIONS_URL[split], cache_path)

    return pd.read_csv(cache_path)


def get_class_mids(class_names: list[str]) -> dict[str, str]:
    """Look up the machine-readable IDs (MIDs) for given class names.

    Returns:
        Dict mapping class_name -> mid
    """
    classes_df = load_class_descriptions()
    found = classes_df[classes_df["class_name"].isin(class_names)]
    return dict(zip(found["class_name"], found["mid"]))


def list_classes_for_category(category: str) -> None:
    """Print the Open Images classes mapped to a project category."""
    if category not in CATEGORY_TO_CLASSES:
        print(f"Unknown category: {category}")
        print(f"Available categories: {list(CATEGORY_TO_CLASSES.keys())}")
        return

    class_names = CATEGORY_TO_CLASSES[category]
    mids = get_class_mids(class_names)

    print(f"\nCategory: {category}")
    print(f"{'Class Name':<30} {'MID':<15} {'Found'}")
    print("-" * 55)
    for name in class_names:
        mid = mids.get(name, "NOT FOUND")
        status = "yes" if name in mids else "MISSING"
        print(f"{name:<30} {mid:<15} {status}")


def find_images_for_category(
    category: str,
    split: str = "train",
    max_images: int = None,
    seed: int = None,
) -> pd.DataFrame:
    """Find image IDs containing objects from a given category.

    Args:
        category: One of the keys in CATEGORY_TO_CLASSES
        split: Dataset split ("train" or "validation")
        max_images: Maximum number of unique images to return
        seed: Random seed for reproducible sampling

    Returns:
        DataFrame with columns: ImageID, OriginalURL, class_name
    """
    if seed is None:
        seed = CONFIG.get("random_seed", 42)

    class_names = CATEGORY_TO_CLASSES[category]
    mids = get_class_mids(class_names)

    if not mids:
        print(f"No matching classes found for category: {category}")
        return pd.DataFrame()

    print(f"Loading {split} annotations...")
    annotations = load_annotations(split)

    # Filter annotations for our target classes
    target_mids = list(mids.values())
    filtered = annotations[annotations["LabelName"].isin(target_mids)]

    # Get unique image IDs
    image_ids = filtered["ImageID"].unique()
    print(f"Found {len(image_ids)} images with {category} objects")

    # Sample if we need fewer
    if max_images and len(image_ids) > max_images:
        rng = pd.np.random.default_rng(seed) if hasattr(pd, "np") else __import__("numpy").random.default_rng(seed)
        image_ids = rng.choice(image_ids, size=max_images, replace=False)
        print(f"Sampled {max_images} images")

    # Get the download URLs
    print("Loading image URLs...")
    image_meta = load_image_ids(split)
    result = image_meta[image_meta["ImageID"].isin(image_ids)][["ImageID", "OriginalURL"]]

    # Add which classes are present
    mid_to_name = {v: k for k, v in mids.items()}
    image_classes = (
        filtered[filtered["ImageID"].isin(image_ids)]
        .groupby("ImageID")["LabelName"]
        .apply(lambda x: ", ".join(sorted(set(mid_to_name.get(m, m) for m in x))))
        .reset_index()
        .rename(columns={"LabelName": "classes_present"})
    )
    result = result.merge(image_classes, on="ImageID", how="left")

    return result


def download_images(
    image_df: pd.DataFrame,
    output_dir: Path,
    skip_existing: bool = True,
) -> dict:
    """Download images from their original URLs.

    Args:
        image_df: DataFrame with ImageID and OriginalURL columns
        output_dir: Directory to save images
        skip_existing: Skip images that already exist on disk

    Returns:
        Dict with download stats: {"downloaded": int, "skipped": int, "failed": int}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    for _, row in tqdm(image_df.iterrows(), total=len(image_df), desc="Downloading"):
        image_id = row["ImageID"]
        url = row["OriginalURL"]
        output_path = output_dir / f"{image_id}.jpg"

        if skip_existing and output_path.exists():
            stats["skipped"] += 1
            continue

        try:
            urllib.request.urlretrieve(url, output_path)
            stats["downloaded"] += 1
        except Exception as e:
            stats["failed"] += 1
            # Log failures but don't stop
            if stats["failed"] <= 10:
                print(f"  Failed: {image_id} - {e}")
            elif stats["failed"] == 11:
                print("  (suppressing further error messages)")

    return stats


def save_manifest(image_df: pd.DataFrame, output_dir: Path, category: str) -> None:
    """Save a CSV manifest of downloaded images for tracking."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"manifest_{category}.csv"
    image_df.to_csv(manifest_path, index=False)
    print(f"Saved manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download real images from Open Images V7 for the NOBLE benchmark."
    )
    parser.add_argument(
        "--category",
        choices=list(CATEGORY_TO_CLASSES.keys()),
        help="Image category to download. If not specified, downloads all categories.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Max images to download. Defaults to config.yaml target for the category.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation"],
        help="Open Images split to use (default: train).",
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="List the Open Images classes mapped to a category and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find images but don't download them. Prints stats and saves manifest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed (default: {CONFIG.get('random_seed', 42)} from config).",
    )

    args = parser.parse_args()

    # List classes mode
    if args.list_classes:
        if args.category:
            list_classes_for_category(args.category)
        else:
            for cat in CATEGORY_TO_CLASSES:
                list_classes_for_category(cat)
        return

    # Determine which categories to download
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

        # Find matching images
        image_df = find_images_for_category(
            category=category,
            split=args.split,
            max_images=max_images,
            seed=args.seed,
        )

        if image_df.empty:
            print(f"No images found for {category}, skipping.")
            continue

        print(f"Found {len(image_df)} images to download")

        # Save manifest regardless
        category_dir = output_base / category
        save_manifest(image_df, category_dir, category)

        if args.dry_run:
            print("Dry run — skipping download.")
            continue

        # Download
        stats = download_images(image_df, category_dir)
        print(f"Results: {stats['downloaded']} downloaded, {stats['skipped']} skipped, {stats['failed']} failed")

    print("\nDone!")


if __name__ == "__main__":
    main()
