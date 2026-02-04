# Team Roles

This project has three student roles, each with distinct responsibilities. Read your role carefully and understand how you'll collaborate with teammates.

---

## Role 1: Data & Generation Lead

**Background:** Computer Science

**Primary responsibilities:**
- Filter and download real images from Open Images and COCO datasets
- Build and run the synthetic image generation pipeline
- Manage batch jobs on TACC
- Ensure dataset quality and organization

### Key tasks by phase

**Weeks 1-2: Setup**
- Set up TACC account and test SSH access
- Understand Open Images and COCO dataset structures
- Review image filtering requirements

**Weeks 3-4: Real image collection**
- Write Python scripts to filter datasets by category labels
- Download ~10,000 real images matching our categories
- Deduplicate using perceptual hashing
- Organize into consistent folder structure

**Weeks 5-6: Synthetic image generation**
- Write prompt list with team input
- Set up diffusers pipeline on TACC
- Run batch generation jobs (SDXL, Flux, SD2.1)
- QC outputs, filter for quality and relevance

**Weeks 7-8: Support augmentation pipeline**
- Help Augmentation Lead apply degradations
- Verify balanced dataset distributions

**Weeks 9-14: Support evaluation and documentation**
- Help run evaluation scripts
- Document data collection process
- Contribute to technical report

### Skills you'll develop
- Working with large-scale image datasets
- HPC job submission and management
- Image generation with diffusion models
- Data pipeline engineering

### Your key files
```
src/data_collection/
src/generation/
data/raw/
```

---

## Role 2: Augmentation & Evaluation Lead

**Background:** Data Science

**Primary responsibilities:**
- Build the image degradation pipeline
- Create all three dataset versions (clean, moderate, heavy)
- Evaluate detection tools and compute metrics
- Produce visualizations and analysis

### Key tasks by phase

**Weeks 1-2: Setup**
- Learn Albumentations library
- Understand evaluation metrics (accuracy, precision, recall, F1, AUC)
- Research detection tool APIs

**Weeks 3-4: Support data collection**
- Help with QC of downloaded images
- Begin documenting augmentation parameters

**Weeks 5-6: Support generation**
- Help review generated image quality
- Begin building augmentation pipeline

**Weeks 7-8: Degradation pipeline**
- Implement all augmentations (JPEG compression, blur, noise, etc.)
- Generate clean, moderate, and heavy dataset versions
- Verify augmentations are applied to both real and synthetic

**Weeks 9-11: Tool evaluation**
- Set up API access for detection tools
- Run all tools on all dataset versions
- Compute metrics, generate confusion matrices
- Create visualizations (AUC curves, per-category breakdowns)

**Weeks 12-14: Analysis and reporting**
- Identify patterns in results
- Produce figures for paper
- Contribute to technical report

### Skills you'll develop
- Image augmentation and preprocessing
- API integration
- Evaluation methodology
- Data visualization and analysis

### Your key files
```
src/augmentation/
src/evaluation/
data/processed/
results/
```

---

## Role 3: Documentation & Analysis Lead

**Background:** Criminal Justice

**Primary responsibilities:**
- Conduct literature review on AI-generated evidence
- Write prompts for synthetic image generation
- Draft reports and policy briefs
- Ensure research is relevant to law enforcement

### Key tasks by phase

**Weeks 1-2: Literature review**
- Research AI-generated evidence in courts (Mendones, Reffitt cases)
- Survey existing detection tools and their claims
- Summarize findings for team

**Weeks 3-4: Prompt development**
- Write 200+ unique prompts for synthetic generation
- Ensure prompts cover all evidence categories
- Make prompts realistic to law enforcement contexts

**Weeks 5-6: Support generation and QC**
- Review generated images for realism and relevance
- Flag problematic outputs
- Refine prompts based on results

**Weeks 7-8: Analysis framework**
- Define what findings would be significant
- Plan policy implications

**Weeks 9-11: Support evaluation**
- Help interpret results from law enforcement perspective
- Identify implications for practice

**Weeks 12-14: Report drafting**
- Draft technical report sections
- Write 2-page policy brief for NOBLE
- Help prepare paper abstract and framing

### Skills you'll develop
- Academic research and literature review
- Technical writing for non-technical audiences
- Policy analysis
- Interdisciplinary collaboration

### Your key files
```
docs/
configs/prompts/
results/reports/
notebooks/portfolios/
```

---

## Collaboration Points

| Phase | Data Lead | Augmentation Lead | Documentation Lead |
|-------|-----------|-------------------|--------------------|
| Weeks 3-4 | Downloading images | QC support | Literature review |
| Weeks 5-6 | Running generation | QC support, pipeline prep | Prompt writing, QC |
| Weeks 7-8 | Dataset support | Running augmentation | Analysis planning |
| Weeks 9-11 | Evaluation support | Running evaluation | Interpretation |
| Weeks 12-14 | Documentation | Figures and metrics | Report writing |

## Shared Responsibilities

All team members:
- Attend weekly meetings
- Document work in portfolio notebooks
- Review teammates' pull requests
- Contribute to final paper
- Present findings to NOBLE

## Questions about your role?

If you're unsure what you should be working on, check:
1. [WEEK1_TASKS.md](WEEK1_TASKS.md) for immediate tasks
2. The timeline in the main project plan
3. Ask Dr. Scruse
