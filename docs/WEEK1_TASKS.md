# Week 1 Tasks

This document outlines specific tasks for each team member during Week 1. Complete all items checked off by end of week.

---

## All Team Members

### Day 1-2: Environment Setup
- [ ] Clone the repository
- [ ] Set up Python environment (conda or venv)
- [ ] Verify all dependencies install correctly
- [ ] Create your portfolio notebook in `notebooks/portfolios/`
- [ ] Read the README and project plan

### Day 3-5: Orientation
- [ ] Read your role description in [ROLES.md](ROLES.md)
- [ ] Read the full Phase 1 section of the project plan
- [ ] Attend the team kickoff meeting
- [ ] Complete your first commit (even if just updating your portfolio notebook)

---

## Data & Generation Lead (CS)

### Day 1-2: Environment + TACC
- [ ] Complete general setup above
- [ ] Request TACC account (if you don't have one)
- [ ] Review TACC documentation: https://docs.tacc.utexas.edu/
- [ ] Set up SSH keys for TACC access

### Day 3-4: Dataset Research
- [ ] Explore the Open Images dataset website
- [ ] Read Open Images download documentation
- [ ] Explore the COCO dataset website
- [ ] Read COCO API documentation
- [ ] Document: What category labels exist that match our needs?

### Day 5: Planning
- [ ] In your portfolio notebook, draft a plan for filtering real images
- [ ] List which category labels you'll use for each image type:
  - People
  - Vehicles
  - Indoor scenes
  - Outdoor scenes
  - Objects
- [ ] Identify potential challenges (file size, download time, etc.)

### Deliverable
By end of Week 1: Portfolio notebook entry with dataset research notes and filtering plan.

---

## Augmentation & Evaluation Lead (Data Science)

### Day 1-2: Environment + Tools
- [ ] Complete general setup above
- [ ] Install and test Albumentations: `python -c "import albumentations"`
- [ ] Review Albumentations documentation: https://albumentations.ai/docs/

### Day 3-4: Augmentation Research
- [ ] In your portfolio notebook, document each augmentation we need:
  - JPEG compression (quality levels: 20, 40, 70)
  - Resolution downscale (25%, 50%)
  - Gaussian blur (σ = 1, 2, 3)
  - Gaussian noise (σ = 10, 25, 50)
  - Salt-pepper noise (1%, 3%, 5%)
  - Contrast reduction (0.5x, 0.7x)
  - Brightness shift (±20%)
- [ ] For each augmentation, find the Albumentations function
- [ ] Test one augmentation on a sample image

### Day 5: Detection Tool Survey
- [ ] Create a list of detection tools to evaluate:
  - Hive Moderation (https://hivemoderation.com/)
  - Illuminarty (https://illuminarty.ai/)
  - AI or Not (https://aiornot.com/)
  - Hugging Face SDXL Detector
  - Others you find
- [ ] Note: Free tier limits, API availability, documentation quality

### Deliverable
By end of Week 1: Portfolio notebook entry with augmentation mapping and detection tool survey.

---

## Documentation & Analysis Lead (Criminal Justice)

### Day 1-2: Environment + Reading
- [ ] Complete general setup above
- [ ] Skim the full project plan, especially Phase 1 and Phase 4

### Day 3-4: Literature Review Start
- [ ] Find and read about the Mendones v. Cushman case
- [ ] Find and read about USA v. Reffitt case
- [ ] Search for: "AI-generated evidence" "deepfake evidence" "synthetic media courts"
- [ ] In your portfolio notebook, start a literature review section:
  - Case summaries (1-2 paragraphs each)
  - Key themes emerging
  - Questions for further research

### Day 5: Prompt Category Planning
- [ ] Review the prompt categories in the project plan:
  - Surveillance/Security (3,000 images)
  - Evidence-style (3,000 images)
  - Bodycam-style (2,500 images)
  - Documents (1,500 images)
- [ ] Brainstorm: What makes these categories realistic to law enforcement?
- [ ] Draft 5 example prompts for each category
- [ ] Note: What details matter? (lighting, angle, quality descriptors)

### Deliverable
By end of Week 1: Portfolio notebook entry with case summaries and initial prompt ideas.

---

## Week 1 Meeting Agenda

**Kickoff Meeting (Day 3 or 4)**

1. Introductions and backgrounds (10 min)
2. Project overview and goals (15 min)
3. Role assignments and questions (15 min)
4. Environment troubleshooting (10 min)
5. Communication norms (10 min)
6. Questions and next steps (10 min)

**End of Week Check-in (Day 5)**

Each person shares:
- What I accomplished
- What I learned
- What I'm stuck on
- Plan for Week 2

---

## Common Week 1 Issues

### "I can't install PyTorch"
Try installing CPU-only version first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### "I don't have a GPU"
That's fine for Week 1. GPU work happens on TACC, not your laptop.

### "I'm not sure if I'm doing this right"
Ask in the team channel. Everyone is learning. There are no dumb questions.

### "I finished early"
- Explore the datasets more deeply
- Read more papers
- Help a teammate
- Get ahead on Week 2 tasks

---

## Success Criteria

By end of Week 1, the team should have:
- [ ] All environments working
- [ ] All portfolio notebooks created
- [ ] First commits from each team member
- [ ] Clear understanding of roles
- [ ] Initial research documented
- [ ] Communication channels established
