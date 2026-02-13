# AI-Generated Evidence Detection Benchmark

**Phase 1: Dataset Creation & Tool Evaluation**

A NOBLE-funded research project to evaluate AI-generated image detection tools on law enforcement content.

## Project Overview

Law enforcement increasingly encounters AI-generated images as potential evidence or as the basis for challenging authentic evidence. While detection tools exist, they are primarily trained and evaluated on social media content—not the grainy, compressed, low-quality images typical of bodycam and surveillance footage.

**Research Questions:**
- How well do existing AI-generated image detection tools perform on law enforcement content?
- How much does detection accuracy degrade on low-quality images typical of bodycam and surveillance footage?

**Deliverables:**
- Benchmark dataset: 20k images (10k real + 10k synthetic) across 3 quality levels
- Tool evaluation report with performance metrics for existing detectors
- Policy brief for NOBLE leadership
- Reproducible code repository

## Team

| Role | Responsibilities |
|------|------------------|
| **PI** | Dr. Ashley Scruse, Morehouse College |
| **Data & Generation Lead** | Real image filtering, synthetic generation pipeline |
| **Augmentation & Evaluation Lead** | Degradation pipeline, tool evaluation scripts |
| **Documentation & Analysis Lead** | Literature review, prompt writing, report drafting |

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ashleyscruse/ai-generated-image-detection.git
cd ai-generated-image-detection
```

### 2. Set up Python environment

```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate aidetect

# OR using pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Verify installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
```

### 4. Read the onboarding guide

Start with [docs/ONBOARDING.md](docs/ONBOARDING.md) for your first week tasks.

## Directory Structure

```
ai-generated-image-detection/
├── data/
│   ├── raw/
│   │   ├── real/           # Downloaded real images from Open Images/COCO
│   │   └── synthetic/      # Generated images from SDXL/Flux/SD2.1
│   └── processed/
│       ├── clean/          # No degradation applied
│       ├── moderate/       # JPEG Q50 + blur + contrast reduction
│       └── heavy/          # JPEG Q30 + downscale + noise + blur
├── src/
│   ├── data_collection/    # Scripts to filter and download real images
│   ├── generation/         # Synthetic image generation pipeline
│   ├── augmentation/       # Degradation and augmentation pipeline
│   ├── evaluation/         # Tool evaluation and metrics
│   └── utils/              # Shared utilities
├── notebooks/
│   ├── exploration/        # Exploratory analysis notebooks
│   └── portfolios/         # Individual student portfolio notebooks
├── docs/                   # Documentation
├── configs/                # Configuration files
└── results/
    ├── metrics/            # Evaluation metrics (CSV, JSON)
    ├── figures/            # Generated plots and visualizations
    └── reports/            # Draft reports and briefs
```

## Documentation

- [Onboarding Guide](docs/ONBOARDING.md) - Start here
- [Team Roles](docs/ROLES.md) - Detailed role descriptions
- [Week 1 Tasks](docs/WEEK1_TASKS.md) - First week checklist
- [Full Project Plan](NOBLE_Full_Project_Plan.md) - Complete 4-phase plan

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data | pandas, Pillow, OpenCV |
| Augmentation | Albumentations |
| Generation | diffusers (HuggingFace), Replicate API |
| Evaluation | scikit-learn, matplotlib, seaborn |
| Notebooks | Jupyter |
| HPC | TACC (Singularity containers) |
| Version Control | Git + GitHub |

## Timeline (14 weeks)

| Week | Focus |
|------|-------|
| 1-2 | Onboarding, environment setup, role assignment |
| 3-4 | Real image filtering and download |
| 5-6 | Synthetic image generation on TACC |
| 7-8 | Degradation pipeline, dataset versions |
| 9-11 | Tool evaluation |
| 12-14 | Report writing, paper draft |

## Contact

**PI:** Dr. Ashley Scruse
**Institution:** Morehouse College, Center for Broadening Participation in Computing
**Funder:** NOBLE (National Organization of Black Law Enforcement Executives)
