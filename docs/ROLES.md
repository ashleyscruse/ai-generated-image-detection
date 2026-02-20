# Team Roles

This project has four students working in parallel across each phase. Rather than each person owning one stage of the pipeline (which creates bottlenecks), everyone works on the same phase together, split by image category or detection tool.

Read the [9-week timeline](TIMELINE_9WEEK.md) to see how your work fits into the full schedule.

---

## How Roles Work in This Project

- **Everyone codes.** Every student writes Python, runs TACC jobs, and produces evaluation results.
- **Everyone writes.** Every student contributes to the final paper and documents their work in a portfolio notebook.
- Roles define your **primary category ownership** and **writing responsibilities**, not a siloed pipeline stage.

---

## CJ Grad Student

**Background:** Criminal Justice (graduate)

**What makes this role unique:** You bring the law enforcement perspective that makes this research meaningful. You'll learn Python, data analysis, and AI tools while ensuring our work is relevant to real policing contexts.

### Your responsibilities across the project

| Phase | Your Work |
|---|---|
| Week 1 | Environment setup, read Mendones + Reffitt case law, explore Open Images + COCO datasets, Python refresher |
| Week 2 | Download Objects + Documents real images (~1,000), lead QC across all categories, write 50 Surveillance/Security prompts |
| Week 3 | Generate Documents synthetic images (~1,500) on TACC, QC outputs for realism |
| Weeks 4-5 | Help build degradation pipeline (paired with a CS student), apply degradation to your images |
| Weeks 6-7 | Run Illuminarty + Optic AI or Not detection tools across full dataset, compute metrics |
| Weeks 8-9 | Draft policy brief for NOBLE + Introduction/Related Work section, review full paper |

### Skills you'll develop
- Python scripting (pandas, API calls, matplotlib)
- Working with HPC (TACC)
- AI image generation with diffusion models
- Evaluation metrics and data visualization
- Academic research and policy writing

### Your key directories
```
All src/ directories (you work across the full pipeline)
docs/
results/reports/
notebooks/portfolios/
```

---

## CS Student 1

**Background:** Computer Science (freshman, Morehouse)

### Your responsibilities across the project

| Phase | Your Work |
|---|---|
| Week 1 | Environment setup, read 1 detection paper, explore Open Images + COCO datasets, help lead Python refresher |
| Week 2 | Download People real images (~2,500), write 50 Evidence-style prompts |
| Week 3 | Generate Surveillance/Security synthetic images (~3,000) on TACC |
| Weeks 4-5 | Finish generation, apply degradation to Surveillance/Security + People images |
| Weeks 6-7 | Run Hive Moderation detection tool across full dataset, compute metrics |
| Weeks 8-9 | Write Dataset section + compile comparison tables, review full paper |

### Skills you'll develop
- Large-scale image dataset management
- HPC job submission and management
- Image generation with diffusion models
- Commercial API integration (rate limiting, authentication)
- Technical writing

### Your key directories
```
src/data_collection/
src/generation/
data/raw/
notebooks/portfolios/
```

---

## CS Student 2

**Background:** Computer Science (freshman, Morehouse)

### Your responsibilities across the project

| Phase | Your Work |
|---|---|
| Week 1 | Environment setup, read 1 detection paper, explore Open Images + COCO datasets, help lead Python refresher |
| Week 2 | Download Vehicles real images (~2,500), write 50 Bodycam-style prompts |
| Week 3 | Generate Evidence-style synthetic images (~3,000) on TACC |
| Weeks 4-5 | Finish generation, apply degradation to Evidence-style + Vehicles images |
| Weeks 6-7 | Run HuggingFace SDXL Detector across full dataset, compute metrics |
| Weeks 8-9 | Write Methodology section + create all figures (AUC curves, confusion matrices), review full paper |

### Skills you'll develop
- Image augmentation and preprocessing
- Open source ML model inference
- Evaluation methodology and visualization
- Data analysis with scikit-learn, matplotlib, seaborn
- Technical writing

### Your key directories
```
src/augmentation/
src/evaluation/
data/processed/
results/figures/
notebooks/portfolios/
```

---

## CS Student 3

**Background:** Computer Science (freshman, Morehouse)

### Your responsibilities across the project

| Phase | Your Work |
|---|---|
| Week 1 | Environment setup, read 1 benchmark paper, explore Open Images + COCO datasets, help lead Python refresher |
| Week 2 | Download Indoor + Outdoor scene real images (~4,000), write 50 Documents prompts |
| Week 3 | Generate Bodycam-style synthetic images (~2,500) on TACC |
| Weeks 4-5 | Finish generation, lead deduplication/dataset organization, apply degradation to Bodycam-style + Indoor/Outdoor images |
| Weeks 6-7 | Run AI or Not + SynthID (if available) detection tools across full dataset, compute metrics |
| Weeks 8-9 | Write Results section + abstract/conclusion, review full paper |

### Skills you'll develop
- Dataset organization and quality control
- API integration (multiple services)
- Statistical analysis and results interpretation
- Technical writing (results communication)
- Data pipeline engineering

### Your key directories
```
src/data_collection/
src/evaluation/
results/metrics/
notebooks/portfolios/
```

---

## Shared Responsibilities

All team members:
- Attend weekly meetings
- Document work in portfolio notebooks
- Review teammates' pull requests
- Contribute to the final paper
- Present findings to NOBLE

---

## Collaboration Points

| Week | What's Happening | How You Help Each Other |
|---|---|---|
| 1 | Onboarding | CS students help CJ Grad with Python/TACC setup |
| 2 | Real image downloads | Everyone uses the same filtering script pattern |
| 3 | Synthetic generation | One person builds the pipeline script, everyone uses it |
| 4-5 | Degradation | Build the pipeline as a team, then each person runs it |
| 6-7 | Tool evaluation | Everyone computes the same metrics, compare notes |
| 8-9 | Writing | Everyone reviews and edits each other's sections |

---

## Questions about your role?

If you're unsure what you should be working on, check:
1. [TIMELINE_9WEEK.md](TIMELINE_9WEEK.md) for the weekly schedule
2. [WEEK1_TASKS.md](WEEK1_TASKS.md) for immediate tasks
3. Ask Dr. Scruse
