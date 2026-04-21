"""Streamlined local generation script optimized for Apple Silicon (M4 Max).

Generates a funder-presentable dataset using all 3 generators with a subset
of prompts. Designed to run overnight or during a work session.

Usage:
    # Generate full funder dataset (~500 images, takes ~2-4 hours on M4 Max)
    python -m src.generation.generate_local

    # Quick test (5 images)
    python -m src.generation.generate_local --test

    # Generate with specific model only
    python -m src.generation.generate_local --model sd21

    # Resume (skips existing)
    python -m src.generation.generate_local --resume
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.generation.generate import load_prompts
from src.utils.config import CONFIG, get_data_path


def get_device():
    """Get best available device for Apple Silicon."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _disable_nsfw(pipe):
    """Disable the NSFW safety checker.

    Our prompts describe law enforcement scenarios (surveillance, evidence photos)
    which trigger false positives. This is a research benchmark -- not user-facing.
    """
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    return pipe


def load_sd15_pipeline(device):
    """Load Stable Diffusion 1.5 -- fast, ungated, good baseline."""
    from diffusers import StableDiffusionPipeline

    print("Loading Stable Diffusion 1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def load_openjourney_pipeline(device):
    """Load OpenJourney v4 -- fine-tuned SD, produces distinct artifacts."""
    from diffusers import StableDiffusionPipeline

    print("Loading OpenJourney v4...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "prompthero/openjourney-v4",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def load_realistic_vision_pipeline(device):
    """Load Realistic Vision 5.1 -- photorealistic outputs, hardest to detect."""
    from diffusers import StableDiffusionPipeline

    print("Loading Realistic Vision 5.1...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


PIPELINE_LOADERS = {
    "sd15": load_sd15_pipeline,
    "openjourney": load_openjourney_pipeline,
    "realistic": load_realistic_vision_pipeline,
}

# Generation params per model
MODEL_PARAMS = {
    "sd15": {"num_inference_steps": 25, "guidance_scale": 7.5, "height": 512, "width": 512},
    "openjourney": {"num_inference_steps": 25, "guidance_scale": 7.5, "height": 512, "width": 512},
    "realistic": {"num_inference_steps": 25, "guidance_scale": 7.5, "height": 512, "width": 512},
}

# How many prompts to sample per category for each model (funder dataset)
FUNDER_SAMPLE = {
    "surveillance_security": 20,
    "evidence_style": 15,
    "bodycam_style": 12,
    "documents": 8,
}

VARIATIONS_PER_PROMPT = 3


def sample_prompts(all_prompts: dict, sample_sizes: dict, seed: int = 42) -> dict:
    """Sample a subset of prompts per category."""
    rng = random.Random(seed)
    sampled = {}
    for cat, prompts in all_prompts.items():
        n = min(sample_sizes.get(cat, 10), len(prompts))
        sampled[cat] = rng.sample(prompts, n)
    return sampled


def generate_with_model(
    model_name: str,
    prompts_by_category: dict,
    num_variations: int,
    output_base: Path,
    seed_start: int = 42,
    resume: bool = True,
) -> dict:
    """Generate images for all categories with one model.

    Returns stats dict.
    """
    device = get_device()
    print(f"\nDevice: {device}")

    loader = PIPELINE_LOADERS[model_name]
    pipe = loader(device)
    params = MODEL_PARAMS[model_name]

    stats = {"generated": 0, "skipped": 0, "failed": 0}

    total = sum(len(p) * num_variations for p in prompts_by_category.values())
    pbar = tqdm(total=total, desc=f"{model_name}")

    for category, prompts in prompts_by_category.items():
        out_dir = output_base / category
        out_dir.mkdir(parents=True, exist_ok=True)

        for p_idx, prompt in enumerate(prompts):
            for v_idx in range(num_variations):
                seed = seed_start + p_idx * num_variations + v_idx
                fname = f"{model_name}_p{p_idx:04d}_v{v_idx:03d}.png"
                fpath = out_dir / fname

                if resume and fpath.exists():
                    stats["skipped"] += 1
                    pbar.update(1)
                    continue

                try:
                    gen = torch.Generator(device=device).manual_seed(seed)

                    image = pipe(
                        prompt=prompt,
                        generator=gen,
                        **params,
                    ).images[0]

                    image.save(fpath)
                    stats["generated"] += 1

                except Exception as e:
                    stats["failed"] += 1
                    if stats["failed"] <= 3:
                        tqdm.write(f"  FAIL [{fname}]: {e}")

                pbar.update(1)

    pbar.close()

    # Free VRAM
    del pipe
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate funder-ready synthetic dataset on Apple Silicon.")
    parser.add_argument("--model", choices=list(PIPELINE_LOADERS.keys()), help="Single model to run.")
    parser.add_argument("--test", action="store_true", help="Quick test: 2 prompts, 1 variation, SD2.1 only.")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip existing images (default: True).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_prompts = load_prompts()
    output_base = get_data_path("raw/synthetic")

    if args.test:
        # Quick test mode
        print("=== TEST MODE ===")
        test_prompts = {cat: ps[:2] for cat, ps in all_prompts.items()}
        stats = generate_with_model("sd15", test_prompts, 1, output_base, args.seed, args.resume)
        print(f"\nTest results: {stats}")
        return

    # Funder dataset
    sampled = sample_prompts(all_prompts, FUNDER_SAMPLE, args.seed)
    total_prompts = sum(len(p) for p in sampled.values())
    total_images = total_prompts * VARIATIONS_PER_PROMPT

    print(f"\n{'='*60}")
    print(f"FUNDER DATASET GENERATION")
    print(f"{'='*60}")
    for cat, ps in sampled.items():
        print(f"  {cat}: {len(ps)} prompts x {VARIATIONS_PER_PROMPT} variations = {len(ps) * VARIATIONS_PER_PROMPT} images")

    models = [args.model] if args.model else ["sd15", "openjourney", "realistic"]

    print(f"\nModels: {models}")
    print(f"Total per model: {total_images} images")
    print(f"Grand total: {total_images * len(models)} images")
    print(f"{'='*60}\n")

    all_stats = {}
    start = time.time()

    for model_name in models:
        print(f"\n--- {model_name.upper()} ---")
        model_start = time.time()

        stats = generate_with_model(
            model_name=model_name,
            prompts_by_category=sampled,
            num_variations=VARIATIONS_PER_PROMPT,
            output_base=output_base,
            seed_start=args.seed,
            resume=args.resume,
        )

        elapsed = time.time() - model_start
        print(f"\n{model_name}: {stats['generated']} generated, "
              f"{stats['skipped']} skipped, {stats['failed']} failed "
              f"({elapsed/60:.1f} min)")
        all_stats[model_name] = stats

    total_elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"COMPLETE in {total_elapsed/60:.1f} min")

    # Save generation summary
    summary = {
        "models": models,
        "prompts_per_category": {cat: len(ps) for cat, ps in sampled.items()},
        "variations_per_prompt": VARIATIONS_PER_PROMPT,
        "stats": all_stats,
        "elapsed_minutes": round(total_elapsed / 60, 1),
    }
    summary_path = output_base / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
