"""Generate synthetic images using diffusers (local/TACC) or Replicate API.

Supports three generators:
  - Stable Diffusion XL (primary)
  - FLUX.1-schnell (secondary)
  - Stable Diffusion 2.1 (tertiary)

Usage:
    # Generate images for one category using SDXL (default)
    python -m src.generation.generate --category surveillance_security --num-per-prompt 10

    # Generate using a specific model
    python -m src.generation.generate --category evidence_style --model flux --num-per-prompt 5

    # Generate all categories with all models
    python -m src.generation.generate --all --num-per-prompt 15

    # Use Replicate API instead of local diffusers (no GPU needed)
    python -m src.generation.generate --category bodycam_style --backend replicate --num-per-prompt 5

    # Dry run: show what would be generated
    python -m src.generation.generate --all --num-per-prompt 10 --dry-run

    # Resume from where you left off (skips existing images)
    python -m src.generation.generate --category documents --num-per-prompt 20 --skip-existing
"""

import argparse
import json
import os
import time
from pathlib import Path

from src.utils.config import CONFIG, get_data_path, get_project_root

# Model short names -> HuggingFace model IDs
MODEL_MAP = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "flux": "black-forest-labs/FLUX.1-schnell",
    "sd21": "stabilityai/stable-diffusion-2-1",
}

# Replicate model versions (for API backend)
REPLICATE_MODELS = {
    "sdxl": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
    "flux": "black-forest-labs/flux-schnell",
    "sd21": "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
}

CATEGORIES = ["surveillance_security", "evidence_style", "bodycam_style", "documents"]


def load_prompts(category: str = None) -> dict[str, list[str]]:
    """Load prompts from the JSON file.

    Args:
        category: If specified, return only prompts for this category.

    Returns:
        Dict mapping category name -> list of prompt strings.
    """
    prompts_path = get_project_root() / "configs" / "prompts" / "prompts.json"

    if not prompts_path.exists():
        raise FileNotFoundError(
            f"Prompts file not found: {prompts_path}\n"
            "Run prompt organization first or create configs/prompts/prompts.json"
        )

    with open(prompts_path) as f:
        all_prompts = json.load(f)

    if category:
        if category not in all_prompts:
            raise ValueError(f"Unknown category: {category}. Available: {list(all_prompts.keys())}")
        return {category: all_prompts[category]}

    return all_prompts


def generate_local(
    prompts: list[str],
    model_name: str,
    output_dir: Path,
    num_per_prompt: int = 10,
    resolution: int = 1024,
    seed_start: int = 42,
    skip_existing: bool = True,
) -> dict:
    """Generate images using local diffusers pipeline (GPU required).

    Args:
        prompts: List of text prompts.
        model_name: Short name (sdxl, flux, sd21).
        output_dir: Directory to save generated images.
        num_per_prompt: Number of images to generate per prompt.
        resolution: Image resolution (width=height).
        seed_start: Starting random seed for reproducibility.
        skip_existing: Skip generation if output file exists.

    Returns:
        Stats dict with counts.
    """
    import torch
    from diffusers import (
        AutoPipelineForText2Image,
        DiffusionPipeline,
        StableDiffusionPipeline,
    )

    model_id = MODEL_MAP[model_name]
    stats = {"generated": 0, "skipped": 0, "failed": 0}

    print(f"Loading model: {model_id}")

    # Load the appropriate pipeline
    if model_name == "sdxl":
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    elif model_name == "flux":
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
    elif model_name == "sd21":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )

    # Move to GPU
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("Using CUDA GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        pipe = pipe.to("mps")
        print("Using Apple MPS")
    else:
        print("WARNING: No GPU detected. Generation will be very slow.")
        pipe = pipe.to("cpu")

    output_dir.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    total = len(prompts) * num_per_prompt
    pbar = tqdm(total=total, desc=f"Generating ({model_name})")

    for prompt_idx, prompt in enumerate(prompts):
        for var_idx in range(num_per_prompt):
            seed = seed_start + prompt_idx * num_per_prompt + var_idx
            filename = f"{model_name}_p{prompt_idx:04d}_v{var_idx:03d}.png"
            output_path = output_dir / filename

            if skip_existing and output_path.exists():
                stats["skipped"] += 1
                pbar.update(1)
                continue

            try:
                generator = torch.Generator(device=pipe.device).manual_seed(seed)

                if model_name == "flux":
                    # FLUX uses different params
                    image = pipe(
                        prompt=prompt,
                        num_inference_steps=4,
                        guidance_scale=0.0,
                        generator=generator,
                        height=resolution,
                        width=resolution,
                    ).images[0]
                else:
                    image = pipe(
                        prompt=prompt,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        generator=generator,
                        height=resolution,
                        width=resolution,
                    ).images[0]

                image.save(output_path)
                stats["generated"] += 1

            except Exception as e:
                stats["failed"] += 1
                if stats["failed"] <= 5:
                    print(f"\n  Failed: {filename} - {e}")

            pbar.update(1)

    pbar.close()

    # Save generation log
    log_path = output_dir / f"generation_log_{model_name}.json"
    log_data = {
        "model": model_id,
        "model_short": model_name,
        "num_prompts": len(prompts),
        "num_per_prompt": num_per_prompt,
        "resolution": resolution,
        "seed_start": seed_start,
        "stats": stats,
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return stats


def generate_replicate(
    prompts: list[str],
    model_name: str,
    output_dir: Path,
    num_per_prompt: int = 10,
    resolution: int = 1024,
    seed_start: int = 42,
    skip_existing: bool = True,
) -> dict:
    """Generate images using the Replicate API (no local GPU needed).

    Requires REPLICATE_API_TOKEN in .env or environment.

    Args:
        prompts: List of text prompts.
        model_name: Short name (sdxl, flux, sd21).
        output_dir: Directory to save generated images.
        num_per_prompt: Number of images per prompt.
        resolution: Image resolution.
        seed_start: Starting seed.
        skip_existing: Skip if file exists.

    Returns:
        Stats dict.
    """
    import urllib.request

    import replicate
    from dotenv import load_dotenv
    from tqdm import tqdm

    load_dotenv(get_project_root() / ".env")

    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise EnvironmentError(
            "REPLICATE_API_TOKEN not found. Set it in .env or environment."
        )

    model_version = REPLICATE_MODELS[model_name]
    stats = {"generated": 0, "skipped": 0, "failed": 0}
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(prompts) * num_per_prompt
    pbar = tqdm(total=total, desc=f"Generating via Replicate ({model_name})")

    for prompt_idx, prompt in enumerate(prompts):
        for var_idx in range(num_per_prompt):
            seed = seed_start + prompt_idx * num_per_prompt + var_idx
            filename = f"{model_name}_p{prompt_idx:04d}_v{var_idx:03d}.png"
            output_path = output_dir / filename

            if skip_existing and output_path.exists():
                stats["skipped"] += 1
                pbar.update(1)
                continue

            try:
                output = replicate.run(
                    model_version,
                    input={
                        "prompt": prompt,
                        "width": resolution,
                        "height": resolution,
                        "seed": seed,
                    },
                )

                # Replicate returns a list of URLs or a FileOutput
                if isinstance(output, list):
                    image_url = str(output[0])
                else:
                    image_url = str(output)

                urllib.request.urlretrieve(image_url, output_path)
                stats["generated"] += 1

            except Exception as e:
                stats["failed"] += 1
                if stats["failed"] <= 5:
                    print(f"\n  Failed: {filename} - {e}")

                # Rate limiting: back off on errors
                time.sleep(1)

            pbar.update(1)

    pbar.close()

    # Save generation log
    log_path = output_dir / f"generation_log_{model_name}_replicate.json"
    log_data = {
        "model": model_version,
        "model_short": model_name,
        "backend": "replicate",
        "num_prompts": len(prompts),
        "num_per_prompt": num_per_prompt,
        "resolution": resolution,
        "seed_start": seed_start,
        "stats": stats,
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic images for the NOBLE benchmark."
    )
    parser.add_argument(
        "--category",
        choices=CATEGORIES,
        help="Prompt category to generate. Use --all for all categories.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate images for all categories.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_MAP.keys()),
        default="sdxl",
        help="Model to use (default: sdxl).",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "replicate"],
        default="local",
        help="Generation backend: local (diffusers + GPU) or replicate (API).",
    )
    parser.add_argument(
        "--num-per-prompt",
        type=int,
        default=10,
        help="Number of images to generate per prompt (default: 10).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=CONFIG["generation"].get("default_resolution", 1024),
        help="Image resolution in pixels (default: from config).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=CONFIG["generation"].get("seed_start", 42),
        help="Starting random seed (default: from config).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip images that already exist (default: True).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without actually generating.",
    )

    args = parser.parse_args()

    if not args.category and not args.all:
        parser.error("Specify --category or --all")

    # Load prompts
    if args.all:
        prompts_by_category = load_prompts()
    else:
        prompts_by_category = load_prompts(args.category)

    # Pick generate function
    generate_fn = generate_replicate if args.backend == "replicate" else generate_local

    output_base = get_data_path("raw/synthetic")

    for category, prompts in prompts_by_category.items():
        print(f"\n{'='*60}")
        print(f"Category: {category} ({len(prompts)} prompts)")
        print(f"Model: {args.model} | Backend: {args.backend}")
        print(f"Images per prompt: {args.num_per_prompt}")
        print(f"Total to generate: {len(prompts) * args.num_per_prompt}")
        print(f"{'='*60}")

        output_dir = output_base / category

        if args.dry_run:
            print("DRY RUN -- would generate to:", output_dir)
            existing = len(list(output_dir.glob("*.png"))) if output_dir.exists() else 0
            print(f"  Existing images: {existing}")
            continue

        stats = generate_fn(
            prompts=prompts,
            model_name=args.model,
            output_dir=output_dir,
            num_per_prompt=args.num_per_prompt,
            resolution=args.resolution,
            seed_start=args.seed_start,
            skip_existing=args.skip_existing,
        )

        print(f"\nResults: {stats['generated']} generated, "
              f"{stats['skipped']} skipped, {stats['failed']} failed")

    print("\nDone!")


if __name__ == "__main__":
    main()
