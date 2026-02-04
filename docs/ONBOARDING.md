# Student Onboarding Guide

Welcome to the NOBLE AI-Generated Evidence Detection project. This guide will help you get set up and ready to contribute.

## Before You Start

Make sure you have:
- [ ] A GitHub account
- [ ] Python 3.10+ installed on your computer
- [ ] Git installed
- [ ] A code editor (VS Code recommended)
- [ ] Access to this repository (ask Dr. Scruse if you don't have it)

## Day 1: Environment Setup

### Step 1: Clone the repository

```bash
git clone <repo-url>
cd ai-generated-image-detection
```

### Step 2: Create your Python environment

**Option A: Using conda (recommended)**
```bash
conda env create -f environment.yml
conda activate aidetect
```

**Option B: Using pip**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Verify your setup

Run these commands to verify everything installed correctly:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import albumentations; print(f'Albumentations: {albumentations.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
```

You should see version numbers printed without errors.

### Step 4: Set up Jupyter kernel

```bash
python -m ipykernel install --user --name aidetect --display-name "AI Detect"
```

### Step 5: Create your portfolio notebook

Create a new notebook in `notebooks/portfolios/` with your name:
```
notebooks/portfolios/firstname_lastname.ipynb
```

This will be your personal workspace to document your contributions.

## Day 2: Understand the Project

### Read these documents (in order)

1. **README.md** - Project overview and structure
2. **NOBLE_Full_Project_Plan.md** - Complete project plan (focus on Phase 1)
3. **docs/ROLES.md** - Find your role and responsibilities
4. **docs/WEEK1_TASKS.md** - Your tasks for this week

### Key concepts to understand

**What we're building:**
A benchmark dataset to test how well AI-generated image detectors work on law enforcement content (bodycam footage, surveillance images, evidence photos).

**Why it matters:**
- AI-generated images can be submitted as fake evidence
- AI-generated images can cast doubt on real evidence ("deepfake defense")
- Current detection tools weren't trained on low-quality law enforcement imagery

**Our approach:**
1. Collect 10,000 real images from public datasets
2. Generate 10,000 synthetic images using AI models
3. Apply degradation (blur, compression, noise) to simulate real-world quality
4. Test existing detection tools on our benchmark
5. Report findings

## Git Workflow

### Before starting work each day

```bash
git pull origin main
```

### Creating a new branch for your work

```bash
git checkout -b feature/your-feature-name
```

### Committing your changes

```bash
git add <files>
git commit -m "Brief description of what you did"
```

### Pushing and creating a pull request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub for review.

### Branch naming conventions

- `feature/` - New functionality (e.g., `feature/real-image-filtering`)
- `fix/` - Bug fixes (e.g., `fix/download-timeout`)
- `docs/` - Documentation (e.g., `docs/prompt-list`)

## Resources Access

### TACC (High-Performance Computing)

You'll need a TACC account for running compute-intensive jobs. Dr. Scruse will:
1. Add you to the project allocation
2. Share TACC onboarding materials
3. Help you set up SSH access

**TACC documentation:** https://docs.tacc.utexas.edu/

### API Keys

Some tools require API keys. Create a `.env` file in the project root (never commit this):

```
# .env - DO NOT COMMIT
REPLICATE_API_TOKEN=your_token_here
HIVE_API_KEY=your_key_here
```

Dr. Scruse will provide shared API keys for the project.

### Datasets

- **Open Images:** https://storage.googleapis.com/openimages/web/index.html
- **COCO:** https://cocodataset.org/

## Communication

### Weekly meetings
- Time: TBD
- Location: TBD
- Come prepared with: progress update, blockers, questions

### Async communication
- Primary: [Slack/Discord/Email - TBD]
- Response time expectation: Within 24 hours on weekdays

### When you're stuck
1. Check the documentation first
2. Search for similar issues on GitHub/Stack Overflow
3. Ask in the team channel
4. Schedule a 1:1 with Dr. Scruse if needed

## Expectations

### Time commitment
- ~10-15 hours per week
- Attend all team meetings
- Meet your weekly milestones

### Quality standards
- All code should include docstrings
- Test your code before pushing
- Document your process in your portfolio notebook
- Ask questions early—don't spin your wheels

### Academic integrity
- This is original research—your contributions matter
- Document your sources
- If you use code from elsewhere, cite it
- You may be a co-author on the resulting paper

## Troubleshooting

### "Module not found" errors
Make sure your virtual environment is activated:
```bash
conda activate aidetect  # or source venv/bin/activate
```

### Git conflicts
```bash
git stash
git pull origin main
git stash pop
# Resolve any conflicts manually
```

### Out of disk space
Large datasets go on external storage or TACC, not your laptop. Ask Dr. Scruse before downloading large files locally.

## Checklist

Complete these items in your first week:

- [ ] Environment set up and verified
- [ ] Repository cloned
- [ ] Portfolio notebook created
- [ ] Read all onboarding documents
- [ ] Attended first team meeting
- [ ] Completed Week 1 tasks for your role
- [ ] Pushed at least one commit

## Questions?

Contact Dr. Ashley Scruse: [email TBD]
