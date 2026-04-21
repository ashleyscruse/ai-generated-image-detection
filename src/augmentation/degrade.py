"""Apply degradation/augmentation to images to simulate real-world law enforcement quality.

Creates three dataset versions from source images:
  - Clean: no degradation (copies originals)
  - Moderate: JPEG Q50 + blur sigma=1 + contrast 0.8x (decent surveillance)
  - Heavy: JPEG Q30 + downscale 50% + noise sigma=25 + blur sigma=2 (poor bodycam/old CCTV)

Usage:
    # Process all images (real + synthetic) into all 3 degradation levels
    python -m src.augmentation.degrade

    # Process only real images
    python -m src.augmentation.degrade --source real

    # Process only synthetic images
    python -m src.augmentation.degrade --source synthetic

    # Process a specific degradation level only
    python -m src.augmentation.degrade --level heavy

    # Dry run: show counts without processing
    python -m src.augmentation.degrade --dry-run

    # Process with custom config overrides
    python -m src.augmentation.degrade --level moderate --jpeg-quality 40 --blur-sigma 1.5
"""

import argparse
import io
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.utils.config import CONFIG, get_data_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Degradation presets from config
DEGRADATION_PRESETS = CONFIG.get("degradation", {
    "clean": {},
    "moderate": {
        "jpeg_quality": 50,
        "blur_sigma": 1.0,
        "contrast_factor": 0.8,
    },
    "heavy": {
        "jpeg_quality": 30,
        "downscale_factor": 0.5,
        "noise_sigma": 25,
        "blur_sigma": 2.0,
    },
})


def find_images(directory: Path) -> list[Path]:
    """Recursively find all image files in a directory."""
    if not directory.exists():
        return []
    return sorted(
        f for f in directory.rglob("*")
        if f.suffix.lower() in IMAGE_EXTENSIONS
        and not f.name.startswith(".")
        and "_duplicates" not in str(f)
    )


def apply_degradation(img: Image.Image, params: dict) -> Image.Image:
    """Apply degradation parameters to a PIL Image.

    Args:
        img: Input PIL Image.
        params: Dict of degradation parameters. Supported keys:
            - jpeg_quality (int): JPEG compression quality (1-100)
            - downscale_factor (float): Resize factor (0.0-1.0)
            - blur_sigma (float): Gaussian blur sigma
            - noise_sigma (float): Gaussian noise standard deviation
            - contrast_factor (float): Contrast multiplier (1.0 = no change)
            - brightness_shift (float): Brightness shift (-1.0 to 1.0)
            - salt_pepper_density (float): Salt-and-pepper noise density

    Returns:
        Degraded PIL Image.
    """
    if not params:
        return img.copy()

    result = img.copy()

    # Ensure RGB
    if result.mode != "RGB":
        result = result.convert("RGB")

    # 1. Downscale (if specified) -- do this first to simulate low-res capture
    if "downscale_factor" in params:
        factor = params["downscale_factor"]
        new_w = max(1, int(result.width * factor))
        new_h = max(1, int(result.height * factor))
        result = result.resize((new_w, new_h), Image.LANCZOS)
        # Scale back up to simulate low-res look at display resolution
        result = result.resize((img.width, img.height), Image.LANCZOS)

    # 2. Contrast adjustment
    if "contrast_factor" in params:
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(params["contrast_factor"])

    # 3. Brightness shift
    if "brightness_shift" in params:
        enhancer = ImageEnhance.Brightness(result)
        # Convert shift to multiplier: 0.2 shift -> 1.2 factor
        result = enhancer.enhance(1.0 + params["brightness_shift"])

    # 4. Gaussian blur
    if "blur_sigma" in params:
        sigma = params["blur_sigma"]
        result = result.filter(ImageFilter.GaussianBlur(radius=sigma))

    # 5. Gaussian noise
    if "noise_sigma" in params:
        arr = np.array(result, dtype=np.float32)
        noise = np.random.normal(0, params["noise_sigma"], arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        result = Image.fromarray(arr)

    # 6. Salt-and-pepper noise
    if "salt_pepper_density" in params:
        arr = np.array(result)
        density = params["salt_pepper_density"]
        # Salt
        salt_mask = np.random.random(arr.shape[:2]) < density / 2
        arr[salt_mask] = 255
        # Pepper
        pepper_mask = np.random.random(arr.shape[:2]) < density / 2
        arr[pepper_mask] = 0
        result = Image.fromarray(arr)

    # 7. JPEG compression (do last -- this is the most common real-world degradation)
    if "jpeg_quality" in params:
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=params["jpeg_quality"])
        buffer.seek(0)
        result = Image.open(buffer).copy()

    return result


def process_directory(
    input_dir: Path,
    output_dir: Path,
    params: dict,
    skip_existing: bool = True,
) -> dict:
    """Apply degradation to all images in a directory tree, preserving structure.

    Args:
        input_dir: Source directory containing images.
        output_dir: Destination directory.
        params: Degradation parameters.
        skip_existing: Skip files that already exist in output.

    Returns:
        Stats dict.
    """
    from tqdm import tqdm

    images = find_images(input_dir)
    stats = {"processed": 0, "skipped": 0, "failed": 0}

    if not images:
        print(f"  No images found in {input_dir}")
        return stats

    for img_path in tqdm(images, desc=f"  {input_dir.name}"):
        # Preserve subdirectory structure
        relative = img_path.relative_to(input_dir)
        output_path = output_dir / relative

        # Always save as JPEG for consistency
        if output_path.suffix.lower() == ".png":
            output_path = output_path.with_suffix(".jpg")

        if skip_existing and output_path.exists():
            stats["skipped"] += 1
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(img_path)

            if not params:
                # Clean level: just copy (convert to JPEG if needed)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(output_path, format="JPEG", quality=95)
            else:
                degraded = apply_degradation(img, params)
                degraded.save(output_path, format="JPEG", quality=95)

            stats["processed"] += 1

        except Exception as e:
            stats["failed"] += 1
            if stats["failed"] <= 5:
                print(f"\n  Failed: {img_path.name} - {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Apply degradation to images for the NOBLE benchmark."
    )
    parser.add_argument(
        "--source",
        choices=["real", "synthetic", "both"],
        default="both",
        help="Which source images to process (default: both).",
    )
    parser.add_argument(
        "--level",
        choices=["clean", "moderate", "heavy", "all"],
        default="all",
        help="Degradation level to apply (default: all).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip images that already exist in output (default: True).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing.",
    )
    # Override params
    parser.add_argument("--jpeg-quality", type=int, help="Override JPEG quality.")
    parser.add_argument("--blur-sigma", type=float, help="Override blur sigma.")
    parser.add_argument("--noise-sigma", type=float, help="Override noise sigma.")
    parser.add_argument("--downscale-factor", type=float, help="Override downscale factor.")
    parser.add_argument("--contrast-factor", type=float, help="Override contrast factor.")
    parser.add_argument("--seed", type=int, default=CONFIG.get("random_seed", 42), help="Random seed.")

    args = parser.parse_args()

    np.random.seed(args.seed)

    # Determine sources
    sources = []
    if args.source in ("real", "both"):
        sources.append(("real", get_data_path("raw/real")))
    if args.source in ("synthetic", "both"):
        sources.append(("synthetic", get_data_path("raw/synthetic")))

    # Determine levels
    if args.level == "all":
        levels = ["clean", "moderate", "heavy"]
    else:
        levels = [args.level]

    for level in levels:
        params = dict(DEGRADATION_PRESETS.get(level, {}))

        # Apply CLI overrides
        if args.jpeg_quality is not None:
            params["jpeg_quality"] = args.jpeg_quality
        if args.blur_sigma is not None:
            params["blur_sigma"] = args.blur_sigma
        if args.noise_sigma is not None:
            params["noise_sigma"] = args.noise_sigma
        if args.downscale_factor is not None:
            params["downscale_factor"] = args.downscale_factor
        if args.contrast_factor is not None:
            params["contrast_factor"] = args.contrast_factor

        for source_name, source_dir in sources:
            output_dir = get_data_path(f"processed/{level}/{source_name}")

            print(f"\n{'='*60}")
            print(f"Level: {level} | Source: {source_name}")
            print(f"Params: {params if params else '(none -- clean copy)'}")
            print(f"Input:  {source_dir}")
            print(f"Output: {output_dir}")

            images = find_images(source_dir)
            print(f"Found {len(images)} source images")

            if args.dry_run:
                existing = len(find_images(output_dir)) if output_dir.exists() else 0
                print(f"Existing in output: {existing}")
                print("DRY RUN -- skipping")
                continue

            stats = process_directory(
                input_dir=source_dir,
                output_dir=output_dir,
                params=params,
                skip_existing=args.skip_existing,
            )

            print(f"Results: {stats['processed']} processed, "
                  f"{stats['skipped']} skipped, {stats['failed']} failed")

    print(f"\n{'='*60}")

    # Save degradation log
    log_path = get_data_path("processed") / "degradation_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_data = {
        "presets": {k: dict(v) if v else {} for k, v in DEGRADATION_PRESETS.items()},
        "seed": args.seed,
        "levels_applied": levels,
        "sources": [s[0] for s in sources],
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
