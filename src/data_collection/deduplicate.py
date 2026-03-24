"""Deduplicate images using perceptual hashing.

Perceptual hashing creates a fingerprint of each image based on its visual
content (not the file bytes). Similar-looking images get similar hashes,
so we can find and remove near-duplicates.

Usage:
    # Check for duplicates in a directory (dry run)
    python -m src.data_collection.deduplicate --input data/raw/real/people --dry-run

    # Remove duplicates (moves them to a _duplicates folder)
    python -m src.data_collection.deduplicate --input data/raw/real/people

    # Check across all real image categories
    python -m src.data_collection.deduplicate --input data/raw/real

    # Adjust sensitivity (lower = stricter matching, default 8)
    python -m src.data_collection.deduplicate --input data/raw/real/people --threshold 5
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

from src.utils.config import get_data_path


def compute_hashes(
    image_dir: Path,
    hash_size: int = 8,
) -> dict[str, str]:
    """Compute perceptual hashes for all images in a directory.

    Args:
        image_dir: Directory containing images
        hash_size: Hash size (higher = more sensitive). Default 8 works well.

    Returns:
        Dict mapping file_path -> hash_string
    """
    hashes = {}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    image_files = [
        f for f in image_dir.rglob("*")
        if f.suffix.lower() in image_extensions and not f.name.startswith(".")
    ]

    for img_path in tqdm(image_files, desc=f"Hashing {image_dir.name}"):
        try:
            img = Image.open(img_path)
            h = imagehash.phash(img, hash_size=hash_size)
            hashes[str(img_path)] = str(h)
        except Exception as e:
            print(f"  Could not hash {img_path.name}: {e}")

    return hashes


def find_duplicates(
    hashes: dict[str, str],
    threshold: int = 8,
) -> list[list[str]]:
    """Find groups of duplicate/near-duplicate images.

    Args:
        hashes: Dict mapping file_path -> hash_string
        threshold: Maximum hash distance to consider as duplicate.
                   0 = exact match only, 8 = fairly similar, 16 = loosely similar.

    Returns:
        List of duplicate groups. Each group is a list of file paths.
        The first item in each group is the "keeper" (alphabetically first).
    """
    # Group by exact hash first (fast)
    hash_to_paths = defaultdict(list)
    for path, h in hashes.items():
        hash_to_paths[h].append(path)

    # Find exact duplicates
    duplicate_groups = []
    for h, paths in hash_to_paths.items():
        if len(paths) > 1:
            duplicate_groups.append(sorted(paths))

    # If threshold > 0, also find near-duplicates
    if threshold > 0:
        hash_items = list(hashes.items())
        seen = set()

        for i in range(len(hash_items)):
            if hash_items[i][0] in seen:
                continue

            group = [hash_items[i][0]]
            h1 = imagehash.hex_to_hash(hash_items[i][1])

            for j in range(i + 1, len(hash_items)):
                if hash_items[j][0] in seen:
                    continue

                h2 = imagehash.hex_to_hash(hash_items[j][1])
                distance = h1 - h2

                if distance <= threshold and distance > 0:
                    group.append(hash_items[j][0])
                    seen.add(hash_items[j][0])

            if len(group) > 1:
                duplicate_groups.append(sorted(group))
                seen.add(hash_items[i][0])

    return duplicate_groups


def remove_duplicates(
    duplicate_groups: list[list[str]],
    dry_run: bool = False,
) -> dict:
    """Remove duplicate images, keeping one from each group.

    Duplicates are moved to a _duplicates subfolder (not deleted).

    Args:
        duplicate_groups: List of duplicate groups from find_duplicates()
        dry_run: If True, just print what would happen

    Returns:
        Stats dict
    """
    stats = {"groups": len(duplicate_groups), "removed": 0, "kept": 0}

    for group in duplicate_groups:
        # Keep the first (alphabetically), remove the rest
        keeper = group[0]
        to_remove = group[1:]

        stats["kept"] += 1
        stats["removed"] += len(to_remove)

        if dry_run:
            print(f"  Keep:   {Path(keeper).name}")
            for r in to_remove:
                print(f"  Remove: {Path(r).name}")
            print()
        else:
            for r in to_remove:
                r_path = Path(r)
                dup_dir = r_path.parent / "_duplicates"
                dup_dir.mkdir(exist_ok=True)
                shutil.move(r, dup_dir / r_path.name)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Find and remove duplicate images using perceptual hashing."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory of images to check. Searches recursively.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=8,
        help="Hash distance threshold (0=exact only, 8=default, 16=loose). Lower = stricter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show duplicates but don't remove them.",
    )

    args = parser.parse_args()
    input_dir = Path(args.input)

    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        return

    print(f"Scanning: {input_dir}")
    print(f"Threshold: {args.threshold} (0=exact, 8=similar, 16=loose)\n")

    # Compute hashes
    hashes = compute_hashes(input_dir)
    print(f"\nHashed {len(hashes)} images")

    # Find duplicates
    groups = find_duplicates(hashes, threshold=args.threshold)

    if not groups:
        print("No duplicates found!")
        return

    total_dupes = sum(len(g) - 1 for g in groups)
    print(f"Found {len(groups)} duplicate groups ({total_dupes} images to remove)\n")

    # Remove or report
    if args.dry_run:
        print("DRY RUN — showing what would be removed:\n")

    stats = remove_duplicates(groups, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\nMoved {stats['removed']} duplicates to _duplicates folders")
        print(f"Kept {stats['kept']} unique images from duplicate groups")
    else:
        print(f"Would remove {stats['removed']} duplicates, keeping {stats['kept']} unique")


if __name__ == "__main__":
    main()
