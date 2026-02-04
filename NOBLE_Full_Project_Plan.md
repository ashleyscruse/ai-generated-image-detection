# NOBLE AI-Generated Evidence Detection Project
## Complete 4-Phase Research Plan

**PI:** Dr. Ashley Scruse, Morehouse College
**Funder:** NOBLE ($25,000)
**Total Duration:** 4 Semesters (2 years)
**Team:** 3 undergraduate students per semester + PI

---

## Executive Summary

This project addresses a critical gap in law enforcement: the inability to reliably detect AI-generated images in evidence contexts. Over four phases, we will build a benchmark dataset, train specialized detection models, develop a practical tool, and produce policy guidance for NOBLE members.

**The problem:** AI-generated images are now realistic enough to be submitted as fake evidence or used to cast doubt on real evidence. Existing detection tools are trained on social media content—not the grainy, compressed, low-quality images typical of bodycam and surveillance footage.

**Our contribution:** Domain-specific detection research for law enforcement, producing both technical tools and actionable policy guidance.

---

## Project Phases Overview

| Phase | Semester | Focus | Primary Output | Paper Output |
|-------|----------|-------|----------------|--------------|
| 1 | 1 | Dataset creation + tool evaluation | Benchmark dataset | Evaluation paper |
| 2 | 2 | Model fine-tuning | Trained detection models | Model paper |
| 3 | 3 | Tool development | Deployable detection tool | System/demo paper |
| 4 | 4 | Policy & training | NOBLE training materials | Policy paper |

---

# Phase 1: Dataset Creation & Tool Evaluation
## Semester 1 (14-16 weeks)

### Objective

Build a law enforcement AI detection benchmark and evaluate how well existing detection tools perform on it.

### Research Question

**Primary:** How well do existing AI-generated image detection tools perform on law enforcement content?

**Secondary:** How much does detection accuracy degrade on low-quality images typical of bodycam and surveillance footage?

---

### Deliverables

| Deliverable | Description |
|-------------|-------------|
| Benchmark Dataset | 20k images (10k real + 10k synthetic), 3 quality levels |
| Tool Evaluation Report | Performance metrics for existing detectors |
| Policy Brief | 2-page actionable summary for NOBLE |
| Code Repository | All scripts, documented and reproducible |
| Student Portfolios | Individual notebooks demonstrating contributions |

---

### Dataset Specifications

#### Real Images (~10,000)

**Source:** Filter Open Images and COCO for law enforcement-relevant categories

| Category | Examples | Estimated Count |
|----------|----------|-----------------|
| People | Full body, partial views, crowds | 2,500 |
| Vehicles | Cars, license plates, parking lots | 2,500 |
| Indoor scenes | Rooms, offices, hallways | 2,000 |
| Outdoor scenes | Streets, buildings, lots | 2,000 |
| Objects | Bags, items on tables, documents | 1,000 |

**Process:**
1. Write Python filtering script using dataset APIs
2. Download images matching category labels
3. Deduplicate using perceptual hashing
4. Manual QC pass to remove unsuitable images
5. Organize into consistent folder structure

---

#### Synthetic Images (~10,000)

**Generators:**
- Stable Diffusion XL (primary)
- Flux (secondary)
- Stable Diffusion 2.1 (tertiary, for artifact diversity)

**Prompt Categories:**

```
SURVEILLANCE/SECURITY (3,000 images)
- "grainy security camera footage of person in parking garage"
- "low quality CCTV screenshot of car at night"
- "blurry surveillance image of storefront entrance"
- "overhead security camera view of hallway"
- "night vision camera footage of backyard"

EVIDENCE-STYLE (3,000 images)
- "photograph of drivers license on table, overhead view"
- "police evidence photo of backpack contents"
- "crime scene photograph of room interior"
- "evidence photo of items on car seat"
- "forensic photograph of shoe print"

BODYCAM-STYLE (2,500 images)
- "bodycam footage screenshot of traffic stop"
- "first person view of hallway, low light"
- "shaky camera photo of vehicle interior"
- "POV shot of person at door, fisheye lens"
- "bodycam perspective of parking lot at night"

DOCUMENTS (1,500 images)
- "photograph of ID card on desk"
- "image of handwritten note on paper"
- "photo of computer screen showing text"
- "picture of receipt on table"
- "photograph of vehicle registration document"
```

**Generation Process:**
1. Create prompt list (200+ unique prompts)
2. Set up batch generation pipeline on TACC
3. Generate 50 variations per prompt
4. Filter outputs for quality and relevance
5. Match category distribution to real images

---

#### Degradation Pipeline

**Purpose:** Simulate real-world law enforcement image quality

**Augmentations (applied to BOTH real and synthetic):**

| Augmentation | Parameters | Simulates |
|--------------|------------|-----------|
| JPEG compression | Quality 20, 40, 70 | Upload/transfer degradation |
| Resolution downscale | 25%, 50% of original | Low-res cameras |
| Gaussian blur | σ = 1, 2, 3 | Motion blur, focus issues |
| Gaussian noise | σ = 10, 25, 50 | Sensor noise |
| Salt-pepper noise | 1%, 3%, 5% | Transmission artifacts |
| Contrast reduction | 0.5x, 0.7x | Poor lighting |
| Brightness shift | ±20% | Over/underexposure |

**Dataset Versions:**

| Version | Augmentations Applied | Simulates |
|---------|----------------------|-----------|
| Clean | None | High-quality digital photos |
| Moderate | JPEG Q50 + blur σ=1 + contrast 0.8x | Decent surveillance footage |
| Heavy | JPEG Q30 + downscale 50% + noise σ=25 + blur σ=2 | Poor bodycam/old CCTV |

---

### Tools to Evaluate

| Tool | Type | Access |
|------|------|--------|
| Hive Moderation | Commercial API | Free tier / paid |
| Illuminarty | Web tool | Free |
| AI or Not | Web tool / API | Free tier |
| Hugging Face SDXL Detector | Open source | Free |
| SynthID Detector | Google model | If available |
| Optic AI or Not | API | Free tier |

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1 (per dataset version)
- AUC-ROC curves
- Confusion matrices
- Confidence calibration plots
- Per-category breakdown
- Per-generator breakdown

---

### Tech Stack (Phase 1)

```
Language:        Python 3.10+
Data handling:   pandas, Pillow, OpenCV
Augmentation:    Albumentations
Generation:      diffusers (HuggingFace), Replicate API
Evaluation:      scikit-learn, matplotlib, seaborn
Notebooks:       Jupyter
HPC:             TACC (Singularity containers)
Version control: Git + GitHub
Tracking:        CSV/JSON logs, optional W&B
```

---

### Timeline (Phase 1)

| Week | Activities | Checkpoint |
|------|------------|------------|
| 1 | Student onboarding, environment setup, GitHub repo | Environments ready |
| 2 | Python workshop, assign roles, begin literature review | Roles assigned |
| 3 | Write real image filtering scripts, test downloads | Script working |
| 4 | Complete real image downloads, organize structure | 10k real images |
| 5 | Write prompt list, set up generation pipeline | Prompts ready |
| 6 | Run batch generation on TACC, QC outputs | 10k synthetic images |
| 7 | Build degradation pipeline, test augmentations | Pipeline working |
| 8 | Generate all 3 dataset versions | 3 versions complete |
| 9 | Set up evaluation framework, test with one tool | Framework ready |
| 10 | Evaluate all tools on clean dataset | Clean results |
| 11 | Evaluate all tools on degraded datasets | All results |
| 12 | Compile results, identify patterns, start draft | Draft findings |
| 13 | Write technical report and policy brief | Draft report |
| 14 | Finalize deliverables, prepare presentation | Complete |

---

### Team Structure (Phase 1)

| Student | Background | Role | Key Tasks |
|---------|------------|------|-----------|
| 1 | CS | Data & Generation Lead | Real image filtering, synthetic generation pipeline |
| 2 | Data Science | Augmentation & Evaluation Lead | Degradation pipeline, tool evaluation scripts |
| 3 | Criminal Justice | Documentation & Analysis Lead | Literature review, prompt writing, report drafting |

---

### HPC Resources (Phase 1)

| Task | GPU Hours |
|------|-----------|
| Synthetic image generation | 50-100 |
| Tool inference (if GPU-based) | 50 |
| Buffer | 50 |
| **Total** | **~150-200** |

---

### Paper Output (Phase 1)

**Title:** "Evaluating AI-Generated Image Detection Tools on Law Enforcement Content: A Benchmark Study"

**Abstract (draft):**
> Law enforcement increasingly encounters AI-generated images as potential evidence or as the basis for challenging authentic evidence. While detection tools exist, they are primarily trained and evaluated on social media content. We present the first benchmark for evaluating AI-generated image detection in law enforcement contexts. Our dataset includes 20,000 images (real and synthetic) across three quality degradation levels simulating bodycam and surveillance footage. We evaluate [N] existing detection tools and find that [key finding]. Our results suggest [implication for practice].

**Target Venues:** FAccT, AIES, NeurIPS Datasets & Benchmarks, CVPR/ICML workshops

---

# Phase 2: Model Fine-Tuning
## Semester 2 (14-16 weeks)

### Objective

Train detection models specifically for law enforcement content and evaluate whether domain-specific fine-tuning improves accuracy on degraded images.

### Research Question

**Primary:** Can we improve AI-generated image detection accuracy by fine-tuning on law enforcement content?

**Secondary:** What visual features do models learn to distinguish real from synthetic law enforcement images?

---

### Deliverables

| Deliverable | Description |
|-------------|-------------|
| Trained Models | 4 architectures fine-tuned on benchmark |
| Model Weights | Published for research use |
| Performance Report | Comparison with Phase 1 baseline tools |
| Interpretability Analysis | Attention maps, failure mode analysis |
| Code Repository | Training pipeline, reproducible |

---

### Models to Train

| Model | Architecture | Why |
|-------|--------------|-----|
| CNNDetection | ResNet-50 based | Established synthetic detection baseline |
| EfficientNet-B4 | CNN | Best accuracy/compute tradeoff |
| ViT-B/16 | Transformer | Different inductive bias than CNNs |
| ConvNeXt-Base | Modern CNN | Strong ImageNet performance |

**Training Configuration:**

```
Base learning rate:  1e-4 (with warmup)
Batch size:          32-64 (depending on GPU memory)
Optimizer:           AdamW
Scheduler:           Cosine annealing
Epochs:              20-50 (early stopping)
Augmentation:        Light (flip, small rotation) during training
Validation split:    80/10/10 train/val/test
```

---

### Training Strategy

**Experiment 1: Baseline**
- Train on clean images only
- Test on all 3 degradation levels
- Establish how much accuracy drops without degradation-aware training

**Experiment 2: Degradation-Aware Training**
- Train on mixed dataset (clean + moderate + heavy)
- Test on all 3 levels
- Hypothesis: Mixed training improves robustness

**Experiment 3: Progressive Training**
- Start with clean, gradually introduce degraded samples
- Curriculum learning approach
- Compare with Experiment 2

**Experiment 4: Per-Generator Analysis**
- Train/test splits stratified by generator (SDXL, Flux, SD2.1)
- Measure cross-generator generalization
- Identify which generators are hardest to detect

---

### Interpretability Analysis

**Methods:**
- GradCAM attention maps (what regions does model focus on?)
- t-SNE visualization of learned embeddings
- Failure case analysis (what images fool the model?)
- Per-category performance breakdown

**Questions to Answer:**
- Do models focus on meaningful artifacts or spurious correlations?
- Which image categories are hardest to classify?
- Do different architectures learn different features?

---

### Tech Stack (Phase 2)

```
Framework:       PyTorch 2.x
Models:          timm library, HuggingFace transformers
Training:        PyTorch Lightning (optional)
Tracking:        Weights & Biases
Interpretability: captum, grad-cam
HPC:             TACC (multi-GPU training)
```

---

### Timeline (Phase 2)

| Week | Activities | Checkpoint |
|------|------------|------------|
| 1 | Student onboarding, review Phase 1 results | Team ready |
| 2 | Set up training infrastructure, test pipelines | Pipeline working |
| 3 | Implement all 4 model architectures | Models ready |
| 4 | Run Experiment 1 (baseline training) | Baseline results |
| 5 | Run Experiment 2 (degradation-aware) | Mixed results |
| 6 | Run Experiment 3 (progressive) | Progressive results |
| 7 | Run Experiment 4 (per-generator) | Generator results |
| 8 | Hyperparameter tuning for best approaches | Optimized models |
| 9 | Begin interpretability analysis | Initial visualizations |
| 10 | Complete interpretability, failure analysis | Analysis complete |
| 11 | Compare with Phase 1 tool baselines | Comparison table |
| 12 | Draft paper, compile results | Draft ready |
| 13 | Revise paper, prepare model release | Paper revised |
| 14 | Finalize, publish models to HuggingFace | Complete |

---

### Team Structure (Phase 2)

| Student | Role | Key Tasks |
|---------|------|-----------|
| 1 | Training Lead | Model implementation, training runs, HPC management |
| 2 | Evaluation Lead | Metrics computation, comparison with Phase 1 |
| 3 | Interpretability Lead | GradCAM analysis, failure cases, visualizations |

---

### HPC Resources (Phase 2)

| Task | GPU Hours |
|------|-----------|
| Model training (4 models × 4 experiments) | 300 |
| Hyperparameter tuning | 100 |
| Inference/evaluation | 50 |
| Interpretability analysis | 50 |
| **Total** | **~500** |

---

### Paper Output (Phase 2)

**Title:** "Improving AI-Generated Image Detection for Law Enforcement Contexts Through Domain-Specific Fine-Tuning"

**Abstract (draft):**
> AI-generated image detectors trained on social media content perform poorly on law enforcement imagery characterized by compression artifacts, low resolution, and sensor noise. Building on our law enforcement detection benchmark, we fine-tune four detection architectures (EfficientNet-B4, ViT-B/16, ConvNeXt, CNNDetection) using degradation-aware training. Our best model achieves [X%] accuracy on heavily degraded images, compared to [Y%] for off-the-shelf detectors. Interpretability analysis reveals that [key finding about what models learn]. We release our trained models to support law enforcement applications.

**Target Venues:** CVPR, ICCV, WACV, IEEE S&P

---

# Phase 3: Tool Development
## Semester 3 (14-16 weeks)

### Objective

Build a practical, deployable detection tool for non-technical law enforcement users.

### Research Question

**Primary:** Can we build a usable detection tool that law enforcement practitioners will trust and adopt?

**Secondary:** What interface design and explanation features increase user trust in detection results?

---

### Deliverables

| Deliverable | Description |
|-------------|-------------|
| LEDetect Tool | Web application for image upload and analysis |
| User Documentation | Guide for law enforcement users |
| API | Programmatic access for integration |
| User Study Results | Feedback from law enforcement partners |
| Open Source Release | GitHub repository with deployment instructions |

---

### Tool Specifications

**Core Features:**

| Feature | Description |
|---------|-------------|
| Image Upload | Drag-and-drop or file select, batch upload support |
| Detection Result | Real/Synthetic classification with confidence score |
| Explanation | Visual heatmap showing suspicious regions |
| Confidence Calibration | "High/Medium/Low confidence" plain-language output |
| Report Generation | PDF report suitable for case files |
| Audit Log | Track all analyses for chain of custody |

**Interface Requirements:**
- No technical knowledge required
- Mobile-friendly (officers in field)
- Works offline (optional desktop version)
- Clear, unambiguous results
- Appropriate uncertainty communication

---

### Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Frontend                            │
│  React/Next.js • Tailwind CSS • Mobile-responsive       │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                      Backend API                         │
│  FastAPI • Python • JWT Auth • Rate Limiting            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   Detection Service                      │
│  PyTorch • Phase 2 Models • GradCAM • GPU Inference     │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                     Storage                              │
│  PostgreSQL (logs) • S3/GCS (images) • Redis (cache)    │
└─────────────────────────────────────────────────────────┘
```

**Deployment Options:**
- Cloud hosted (for demo/evaluation)
- On-premise Docker deployment (for departments with data sensitivity)
- Standalone desktop app (Electron, for offline use)

---

### User Study Design

**Participants:** 10-15 law enforcement personnel (through NOBLE/Brookhaven PD contacts)

**Protocol:**
1. Brief introduction to tool (5 min)
2. Guided tasks: analyze 10 images, make judgments (20 min)
3. Open exploration (10 min)
4. Semi-structured interview (15 min)
5. SUS usability questionnaire

**Metrics:**
- Task completion rate
- Time to decision
- Trust in results (self-reported)
- System Usability Scale score
- Qualitative feedback themes

**IRB:** Required for user study—submit early in semester

---

### Tech Stack (Phase 3)

```
Frontend:        Next.js, React, Tailwind CSS
Backend:         FastAPI, Python
Database:        PostgreSQL, Redis
ML Serving:      PyTorch, ONNX (for optimization)
Deployment:      Docker, Google Cloud Run / AWS
Auth:            JWT, optional SSO
Monitoring:      Sentry, basic analytics
```

---

### Timeline (Phase 3)

| Week | Activities | Checkpoint |
|------|------------|------------|
| 1 | Student onboarding, review Phase 2 models | Team ready |
| 2 | Design UI mockups, user flow diagrams | Designs approved |
| 3 | Set up project scaffold, CI/CD | Infrastructure ready |
| 4 | Implement backend API, model serving | API working |
| 5 | Implement frontend, basic upload flow | Upload working |
| 6 | Add explanation features (heatmaps) | Explanations working |
| 7 | Add report generation, audit logging | Features complete |
| 8 | Internal testing, bug fixes | Beta ready |
| 9 | Submit IRB, recruit user study participants | IRB submitted |
| 10 | Deploy to cloud, prepare user study materials | Deployed |
| 11 | Conduct user study sessions | Study complete |
| 12 | Analyze user study data, iterate on feedback | Analysis done |
| 13 | Final polish, write documentation | Docs complete |
| 14 | Open source release, paper draft | Complete |

---

### Team Structure (Phase 3)

| Student | Role | Key Tasks |
|---------|------|-----------|
| 1 | Backend Lead | API, model serving, database, deployment |
| 2 | Frontend Lead | UI implementation, user experience |
| 3 | User Research Lead | Study design, participant recruitment, analysis |

---

### HPC Resources (Phase 3)

| Task | GPU Hours |
|------|-----------|
| Model optimization (ONNX conversion) | 20 |
| Inference during user study | 30 |
| **Total** | **~50** |

*Note: Most Phase 3 work is software development, not compute-intensive.*

---

### Paper Output (Phase 3)

**Title:** "LEDetect: A Practical Tool for AI-Generated Evidence Detection in Law Enforcement"

**Abstract (draft):**
> Despite advances in AI-generated image detection, no tools are designed for law enforcement practitioners who lack technical expertise. We present LEDetect, an open-source web application that allows officers and prosecutors to upload images and receive authenticity assessments with visual explanations. Our tool integrates detection models fine-tuned for law enforcement contexts (bodycam, surveillance footage) and provides calibrated confidence scores. A user study with [N] law enforcement personnel found [key usability findings]. We release LEDetect as open-source software with deployment documentation for departments.

**Target Venues:** CHI, CSCW, UIST, demo tracks

---

# Phase 4: Policy & Training Materials
## Semester 4 (14-16 weeks)

### Objective

Translate technical findings into actionable policy guidance and training materials for NOBLE members.

### Research Question

**Primary:** How should courts and law enforcement departments handle AI-generated evidence based on what detection can and cannot reliably do?

**Secondary:** What training do officers and prosecutors need to appropriately use detection tools?

---

### Deliverables

| Deliverable | Description |
|-------------|-------------|
| Policy White Paper | Comprehensive guidance for NOBLE leadership |
| Training Curriculum | Workshop materials for departments |
| Quick Reference Guide | 1-page field guide for officers |
| Video Training Module | 20-min overview for department training |
| Rule 707 Comment | Formal submission to Federal Rules Committee |

---

### Policy White Paper Outline

**1. Executive Summary** (2 pages)
- Key findings in plain language
- Top 5 recommendations

**2. The Problem** (5 pages)
- AI-generated evidence landscape
- Case law review (Mendones, Reffitt, etc.)
- The "Liar's Dividend" problem

**3. Technical Realities** (10 pages)
- What detection can do (our Phase 1-2 findings)
- What detection cannot do (limitations, failure modes)
- Why "100% accurate" detection is impossible
- Quality degradation effects (bodycam, surveillance)

**4. Current Legal Framework** (5 pages)
- Authentication standards (FRE 901)
- Proposed Rule 707 analysis
- State-level variations

**5. Recommendations** (10 pages)
- For officers collecting evidence
- For prosecutors presenting evidence
- For defense challenging evidence
- For judges evaluating challenges
- For departments adopting detection tools

**6. Implementation Roadmap** (5 pages)
- Training requirements
- Tool deployment considerations
- Documentation practices

---

### Training Curriculum

**Module 1: Understanding AI-Generated Content** (45 min)
- What is generative AI?
- Types of synthetic content (images, text, audio, video)
- How generation technology works (conceptual)
- Live demo of image generation

**Module 2: Detection Tools and Limitations** (45 min)
- How detection works (conceptual)
- Hands-on with LEDetect tool
- Understanding confidence scores
- When to trust/not trust results

**Module 3: Evidence Handling Best Practices** (45 min)
- Documentation requirements
- Chain of custody for digital evidence
- When to seek expert analysis
- Courtroom testimony preparation

**Module 4: Case Studies** (45 min)
- Mendones v. Cushman analysis
- USA v. Reffitt analysis
- Hypothetical scenarios with discussion

**Materials:**
- Slide decks (branded, accessible)
- Handouts with key points
- Assessment quiz
- Facilitator guide for department trainers

---

### Practitioner Research

**Interviews:** 10-15 law enforcement personnel
- Current practices for evidence authentication
- Awareness of AI-generated content threats
- Training needs and preferences
- Barriers to adopting new tools

**Survey:** Broader distribution through NOBLE
- Quantitative data on awareness, practices, needs
- Validate interview findings

**IRB:** Required—can potentially piggyback on Phase 3 approval

---

### Rule 707 Comment

**Structure:**
1. Introduction and credentials
2. Summary of research findings
3. Technical analysis of proposed rule
4. Recommendations for strengthening
5. Practical implementation considerations

**Timeline:** Comment period closes Feb 2026—aim for submission by end of Phase 4

---

### Tech Stack (Phase 4)

```
Documents:       Google Docs, Markdown
Slides:          Google Slides, PowerPoint
Video:           Camtasia or similar
Survey:          Qualtrics
Analysis:        NVivo (qualitative), R/Python (quantitative)
Design:          Canva (quick reference guide)
```

---

### Timeline (Phase 4)

| Week | Activities | Checkpoint |
|------|------------|------------|
| 1 | Student onboarding, review all prior findings | Team ready |
| 2 | Literature review: legal frameworks, case law | Review complete |
| 3 | Design interview protocol, submit IRB | IRB submitted |
| 4 | Begin practitioner interviews | Interviews started |
| 5 | Complete interviews, begin survey design | Interviews done |
| 6 | Deploy survey, begin white paper draft | Survey live |
| 7 | Close survey, analyze qualitative data | Data collected |
| 8 | Draft white paper sections 1-3 | Draft started |
| 9 | Draft white paper sections 4-6 | Draft complete |
| 10 | Develop training curriculum outline | Curriculum outlined |
| 11 | Create training slides and materials | Materials drafted |
| 12 | Record video module, create quick reference | Media complete |
| 13 | Draft Rule 707 comment | Comment drafted |
| 14 | Finalize all deliverables, NOBLE presentation | Complete |

---

### Team Structure (Phase 4)

| Student | Background | Role | Key Tasks |
|---------|------------|------|-----------|
| 1 | Criminal Justice | Policy Lead | White paper drafting, legal analysis |
| 2 | Data Science | Research Lead | Interviews, survey, data analysis |
| 3 | CS | Materials Lead | Training slides, video, quick reference |

**Co-authorship:** Professor Elycia Daniels (Criminal Justice) as co-author on policy paper

---

### Paper Output (Phase 4)

**Title:** "AI-Generated Evidence in the Courtroom: Technical Realities and Policy Recommendations"

**Abstract (draft):**
> As AI-generated images become increasingly realistic, courts face new challenges in evidence authentication. Drawing on three semesters of technical research—including a detection benchmark, fine-tuned models, and a deployed tool—we bridge technical findings with policy implications. Through interviews with [N] law enforcement practitioners and analysis of emerging legal frameworks, we identify gaps between detection capabilities and courtroom needs. We propose evidence handling guidelines, training recommendations, and specific suggestions for the Federal Rules Advisory Committee's proposed Rule 707. Our findings suggest that [key policy insight].

**Target Venues:** Law reviews, FAccT, CSCW, Policy & Internet, criminology journals

---

# Cross-Phase Elements

## Budget Summary (All Phases)

| Item | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Total |
|------|---------|---------|---------|---------|-------|
| Student support | Per grant | Per grant | Per grant | Per grant | $25,000 |
| TACC compute | Included | Included | Included | N/A | $0 |
| Cloud hosting | N/A | N/A | ~$100 | N/A | ~$100 |
| API costs | ~$50 | N/A | N/A | N/A | ~$50 |
| Survey tools | N/A | N/A | N/A | ~$50 | ~$50 |
| **Total additional** | | | | | **~$200** |

---

## HPC Resources (All Phases)

| Phase | GPU Hours | Primary Use |
|-------|-----------|-------------|
| 1 | 150-200 | Image generation |
| 2 | 500 | Model training |
| 3 | 50 | Model optimization |
| 4 | 0 | N/A |
| **Total** | **~750** | |

---

## Publication Timeline

| Phase | Paper | Draft Complete | Target Submission | Target Venue |
|-------|-------|----------------|-------------------|--------------|
| 1 | Benchmark evaluation | End Sem 1 | Summer/Fall | FAccT, AIES, NeurIPS D&B |
| 2 | Model fine-tuning | End Sem 2 | Following cycle | CVPR, ICCV, WACV |
| 3 | System/demo | End Sem 3 | Following cycle | CHI, CSCW |
| 4 | Policy | End Sem 4 | Following cycle | Law review, FAccT |

---

## Risk Mitigation (All Phases)

| Risk | Phase | Mitigation |
|------|-------|------------|
| Generation quality issues | 1 | Multiple generators, QC step |
| Training instability | 2 | Pretrained weights, proven architectures |
| Deployment complexity | 3 | Docker containerization, cloud fallback |
| IRB delays | 3, 4 | Submit early, have backup analysis plan |
| Student turnover | All | Documentation, knowledge transfer sessions |
| TACC access issues | 1, 2 | Apply early, have local backup for testing |

---

## Success Metrics (Full Project)

| Metric | Target |
|--------|--------|
| Benchmark dataset size | 20k+ images |
| Detection tool evaluation | 5+ tools tested |
| Model accuracy improvement | >10% over baseline on degraded images |
| LEDetect users (pilot) | 10+ law enforcement personnel |
| User study SUS score | >70 (acceptable usability) |
| NOBLE presentation | Delivered |
| Papers submitted | 4 (one per phase) |
| Rule 707 comment | Submitted |
| Students trained | 12 (3 per semester × 4 semesters) |

---

## Contact

**PI:** Dr. Ashley Scruse
**Institution:** Morehouse College, Department of Computer Science
**Center:** Center for Broadening Participation in Computing
**Partners:** TACC, NOBLE, Brookhaven Police Department

---

## Appendices

### A. Key References

1. Wang et al. (2020). "CNN-generated images are surprisingly easy to spot...for now." CVPR.
2. Corvi et al. (2023). "On the detection of synthetic images generated by diffusion models." ICASSP.
3. NIST (2025). "Guardians of Forensic Evidence: Evaluating Analytic Systems Against AI-Generated Deepfakes."
4. Federal Rules Advisory Committee. Proposed Rule 707: Authenticating Digital Evidence.

### B. Related Case Law

- Mendones v. Cushman (CA) — 9 deepfake evidence pieces, case dismissed
- USA v. Reffitt (D.C.) — False deepfake defense claim, convicted
- USA v. Khalilian — Voice recording authenticity challenge

### C. Dataset Category Taxonomy

[To be developed in Phase 1]

### D. Model Architecture Details

[To be developed in Phase 2]

### E. LEDetect User Guide

[To be developed in Phase 3]

### F. Training Facilitator Guide

[To be developed in Phase 4]
