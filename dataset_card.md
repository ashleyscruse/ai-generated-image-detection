---
license: cc-by-nc-4.0
task_categories:
  - image-classification
tags:
  - ai-detection
  - synthetic-image-detection
  - law-enforcement
  - benchmark
  - image-forensics
  - deepfake-detection
size_categories:
  - 10K<n<100K
---

# NOBLE AI-Generated Evidence Detection Benchmark

A domain-specific benchmark for evaluating AI-generated image detection tools on law enforcement imagery (surveillance footage, bodycam, evidence-style photos). The benchmark spans multiple generator architectures and three image-quality levels designed to mimic the conditions in which real evidence reaches courtrooms.

**Status:** v1.1 release. All three generators (FLUX-schnell, Realistic Vision 5.1, SDXL) complete, paired-prompt design realized across the full benchmark.

## Why this dataset exists

Existing AI image detection tools were trained on social media imagery: clean, well-lit, high-resolution. The reality of law enforcement footage is none of those things. Bodycam, dashcam, and surveillance video are typically grainy, compressed, and recorded under variable lighting. Detection tools that work on social media content can fail dramatically on the very imagery courts must evaluate when AI-generated material is presented as evidence.

This benchmark fills that gap. It is the first dataset designed specifically to test AI-image detectors on degraded, domain-matched law enforcement content.

## Dataset composition (v1)

### Real images

| Source | Count | Notes |
|---|---:|---|
| UCF Crime surveillance frames | 7,746 | Stratified sample across 14 anomaly + normal categories |
| **Total real** | **7,746** | |

Real surveillance frames are sampled from the UCF Crime Dataset (Sultani et al., 2018). Each category contributes approximately 360 frames, with NormalVideos providing twice that count to support a non-incident baseline.

### Real image categories

| Category | Count | Source |
|---|---:|---|
| Train/Abuse | 336 | UCF Crime Train |
| Train/Arrest | 360 | UCF Crime Train |
| Train/Arson | 328 | UCF Crime Train |
| Train/Assault | 329 | UCF Crime Train |
| Train/Burglary | 348 | UCF Crime Train |
| Train/Explosion | 348 | UCF Crime Train |
| Train/Fighting | 360 | UCF Crime Train |
| Test/Abuse | 297 | UCF Crime Test |
| Test/Arrest | 360 | UCF Crime Test |
| Test/Arson | 360 | UCF Crime Test |
| Test/Assault | 360 | UCF Crime Test |
| Test/Burglary | 360 | UCF Crime Test |
| Test/Explosion | 360 | UCF Crime Test |
| Test/Fighting | 360 | UCF Crime Test |
| Test/Robbery | 360 | UCF Crime Test |
| Test/Shooting | 360 | UCF Crime Test |
| Test/Shoplifting | 360 | UCF Crime Test |
| Test/Stealing | 360 | UCF Crime Test |
| Test/Vandalism | 360 | UCF Crime Test |
| Test/RoadAccidents | 360 | UCF Crime Test |
| NormalVideos | 720 | UCF Crime baseline |
| **Total** | **7,746** | |

### Synthetic images

| Generator | Count | Architecture | Model ID |
|---|---:|---|---|
| Realistic Vision 5.1 | 3,600 | SD 1.5 fine-tune (UNet) | SG161222/Realistic_Vision_V5.1_noVAE |
| SDXL | 3,600 | UNet, multi-stage | stabilityai/stable-diffusion-xl-base-1.0 |
| FLUX.1-schnell | 3,600 | DiT (transformer) | black-forest-labs/FLUX.1-schnell |
| **Total synthetic** | **10,800** | | |

Each generator was run with the same 240 prompts (paired-prompt design), 15 variations per prompt, distributed as: surveillance_security (1,500), evidence_style (900), bodycam_style (750), documents (450). All three generators use matched seeds per prompt, enabling paired statistical comparison.

### Degradation levels

Each image is processed at three quality levels to simulate real-world law enforcement capture conditions:

| Level | Parameters | Simulates |
|---|---|---|
| Clean | None | High-quality digital photos |
| Moderate | JPEG Q50, blur sigma 1.0, contrast 0.8x | Decent surveillance footage |
| Heavy | JPEG Q30, downscale 0.5x, noise sigma 25, blur sigma 2.0 | Poor bodycam, old CCTV |

### Dataset size summary

| Split | Count |
|---|---:|
| Raw real | 7,746 |
| Raw synthetic | 10,800 |
| Processed real (clean / moderate / heavy) | 23,238 |
| Processed synthetic (clean / moderate / heavy × 3 generators) | 32,400 |
| **Total image instances** | **74,184** |

## Dataset structure

```
data/
  raw/
    real/
      surveillance/
        train_<category>/
        test_<category>/
        normal_videos/
    synthetic/
      surveillance_security/
      evidence_style/
      bodycam_style/
      documents/
  processed/
    clean/
      real/
      synthetic/
    moderate/
      real/
      synthetic/
    heavy/
      real/
      synthetic/
```

File naming: synthetic images use the convention `<model>_p<prompt_id>_v<variation_id>.png` (e.g., `rv51_p0042_v003.png`), enabling matched-pair comparison across generators.

## Preliminary results

Pilot evaluation of the HuggingFace AI image detector (`umm-maybe/AI-image-detector`) on a smaller earlier version of this benchmark:

| Quality Level | Accuracy | F1 | AUC-ROC |
|---|---:|---:|---:|
| Clean | 44.5% | 37.4% | 0.402 |
| Moderate | 45.1% | 17.1% | 0.381 |
| Heavy | 34.5% | 18.2% | 0.265 |

The detector performs **worse than random chance** on degraded law enforcement content, with accuracy degrading further as image quality decreases. Full evaluation across multiple detection tools and the v1 benchmark is in progress.

## Intended uses

- Benchmarking AI-generated image detection tools on domain-specific content
- Studying the effect of image degradation on detection accuracy
- Studying generalization gaps across generator architectures (UNet vs. DiT, when FLUX is added)
- Training and fine-tuning improved detection models for law enforcement contexts
- Informing policy discussions on AI evidence admissibility (e.g., Federal Rule 707)

## Out-of-scope uses

- Real-time evidence authentication without further calibration on operational data
- Standalone forensic verification of disputed evidence in legal proceedings
- Training generative models intended to fabricate law enforcement imagery

## Licensing

The benchmark structure, organization, metadata, and associated code are released under **CC-BY-NC-4.0** (Creative Commons Attribution-NonCommercial 4.0 International).

This dataset is compositional. Downstream users must respect the licenses of source materials:

- UCF Crime videos: research-use license (Sultani et al., 2018)
- Realistic Vision 5.1 outputs: CreativeML Open RAIL-M
- SDXL outputs: CreativeML Open RAIL++
- FLUX.1-schnell outputs (when added): Apache 2.0
- Generation prompts: CC-BY-NC-4.0 with this benchmark

Commercial deployment requires separate evaluation against each source material's terms.

## Citation

If you use this benchmark, please cite:

```bibtex
@dataset{scruse2026noble,
  author    = {Scruse, Ashley},
  title     = {NOBLE AI-Generated Evidence Detection Benchmark},
  year      = {2026},
  publisher = {HuggingFace},
  version   = {1.0}
}
```

## Funding and compute

Funded by the National Organization of Black Law Enforcement Executives (NOBLE) through a research grant to Morehouse College. Compute powered by the Texas Advanced Computing Center (TACC) on Vista and Stampede3 systems.

## Acknowledgments

This work was conducted at the Center for Broadening Participation in Computing, Morehouse College, with assistance from a student research team contributing to literature review, dataset exploration, and project documentation.

## Contact

**Ashley Scruse, PhD**
Postdoctoral Researcher, Center for Broadening Participation in Computing, Morehouse College
ashley.scruse@morehouse.edu

## Version history

- **v1.0** (2026-05-06): Initial release. Real surveillance set (7,746) + synthetic from Realistic Vision 5.1 (3,600) and SDXL (3,600). Three degradation levels applied across all images.
- **v1.1** (2026-05-11): Addition of FLUX.1-schnell synthetic set (3,600 images). Full paired-prompt design realized across all three generators (matched seeds per prompt), enabling cross-architecture (DiT vs. UNet) detection evaluation. Total: 74,184 image instances.
