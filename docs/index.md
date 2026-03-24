---
layout: default
title: Student Guide
---

# NOBLE Research: Student Guide

This guide walks you through everything you need to work on the project. Bookmark this page. Come back whenever you're stuck.

---

## 1. Clone the Repo

Open a terminal (Mac: Terminal app, Windows: PowerShell or Git Bash) and run:

```bash
git clone https://github.com/ashleyscruse/ai-generated-image-detection.git
cd ai-generated-image-detection
```

> **Don't have Git?** Install it from [git-scm.com/downloads](https://git-scm.com/downloads).

---

## 2. Set Up Your Environment

### Create a virtual environment

```bash
python3 -m venv venv
```

### Activate it

**Mac/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your terminal prompt. This means it's active.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Register the Jupyter kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name aidetect --display-name "AI Detect (NOBLE)"
```

### Open your notebook

Open VS Code, then open your notebook at:

```
notebooks/portfolios/your_name.ipynb
```

When it asks you to select a kernel, choose **"AI Detect (NOBLE)"**.

---

## 3. Working with Git

You'll be working on **your own branch**. This keeps your work separate from everyone else's.

### Switch to your branch

```bash
git checkout your-branch-name
```

Dr. Scruse will tell you your branch name. It will be something like `tigris`, `roland`, `brandyn`, `nathan`, or `michael`.

### Pull the latest changes

Before you start working each time, pull any updates:

```bash
git pull origin main
```

### Save your work (commit + push)

After making changes to your notebook:

```bash
# Stage your changes
git add notebooks/portfolios/your_name.ipynb
git add data/prompts/

# Commit with a message about what you did
git commit -m "Added 5 prompts and exploration notes"

# Push to YOUR branch
git push origin your-branch-name
```

### Rules

- **Always push to your own branch**, never to `main`
- **Pull from `main`** to get updates from Dr. Scruse
- **Commit often** -- don't wait until everything is done
- Write short, clear commit messages

### If you get a merge conflict

Don't panic. Tell Dr. Scruse. She'll help you resolve it.

---

## 4. How to Use Your Notebook

Your notebook has 8 parts. Here's what to do with each:

| Part | What's Inside | What You Do |
|------|--------------|-------------|
| **Part 0: Setup** | Imports and paths | Just run it |
| **Part 1: Explore** | Browse Open Images dataset | Run it, then answer the questions |
| **Part 2: Download** | Download real images for your category | Run it, then answer the questions |
| **Part 3: Prompts** | Example prompts + space for yours | Study examples, then write your own |
| **Part 4: Generate** | AI image generation (needs GPU) | Run on TACC when ready |
| **Part 5: Degrade** | Apply quality degradation | Run it, then answer the questions |
| **Part 6: Readings** | Space for reading summaries | Fill in your assigned readings |
| **Part 7: Progress** | Progress tracker | Run it to see where you stand |

**Two types of cells:**
- **Code cells** -- click the play button to run them. The code works; you don't need to change it.
- **"YOUR TURN" sections** -- markdown cells where you write observations, prompts, and reading notes. Double-click to edit, then click away to save.

---

## 5. Storing Images

Images are **not stored in Git**. They are too large. The `.gitignore` file prevents them from being committed.

When you download or generate images, they go to:

```
data/
  raw/
    real/         # Downloaded real images (your category folder)
    synthetic/    # Generated images (your category folder)
  processed/
    clean/        # Original quality
    moderate/     # Surveillance quality
    heavy/        # Bad bodycam quality
```

These folders exist on your machine (and on TACC) but are **not uploaded to GitHub**. This is normal.

**What IS tracked in Git:**
- Your notebook (`notebooks/portfolios/your_name.ipynb`)
- Your prompts (`data/prompts/your_prompts.json`)
- Manifests (logs of what you downloaded)

**What is NOT tracked:**
- Image files (`.jpg`, `.png`)
- Model weights
- Large CSV files

> If you switch machines (laptop to TACC), you'll need to re-download images on the new machine. That's expected.

---

## 6. Accessing TACC

We use the **Texas Advanced Computing Center (TACC)** for GPU-heavy work like generating synthetic images. Dr. Scruse will add you to the allocation.

### Step 1: Create Your TACC Account

You'll receive an email from TACC with the subject:

> **"TACC Project Invitation Action Required: Account Request"**

1. Click the link in the email to create your account
2. Use your **institutional email** (e.g., @morehouse.edu, @cau.edu)
3. Choose a username you'll remember

> **Important:** Use the link in the invitation email. Don't create an account separately.

### Step 2: Set Up Multi-Factor Authentication (MFA)

TACC requires MFA. Set this up **before** trying to log in.

> **Do NOT use SMS/text for MFA.** Use an authenticator app.

1. Go to [accounts.tacc.utexas.edu](https://accounts.tacc.utexas.edu/)
2. Click **Multi-factor Auth** in the sidebar
3. Set up one of these apps:
   - **Okta Verify** (recommended) -- [iOS](https://apps.apple.com/app/okta-verify/id490179405) / [Android](https://play.google.com/store/apps/details?id=com.okta.android.auth)
   - **Duo Mobile** -- [iOS](https://apps.apple.com/app/duo-mobile/id422663827) / [Android](https://play.google.com/store/apps/details?id=com.duosecurity.duomobile)

### Step 3: Log In via SSH

```bash
ssh your_username@vista.tacc.utexas.edu
```

When prompted:
1. Enter your TACC password
2. Enter your MFA token (from the authenticator app)

> Nothing appears on screen when you type your password. That's normal. Just type and press Enter.

**Windows users:** Use [PuTTY](https://www.putty.org/) or Windows PowerShell.

### Step 4: Using the TACC Analysis Portal (Jupyter)

For this project, you can also use TACC's web-based Jupyter interface instead of SSH:

1. Go to [tap.tacc.utexas.edu](https://tap.tacc.utexas.edu/)
2. Log in with your TACC credentials
3. Start a Jupyter session on Vista
4. Upload or clone the repo there
5. Open your notebook and run the GPU cells (Part 4)

### TACC File System

| Location | What to put there | Size limit |
|----------|------------------|-----------|
| `$HOME` | Config files, scripts | ~10 GB |
| `$WORK` | Code, datasets, this repo | ~1 TB |
| `$SCRATCH` | Job output, temp files | Unlimited (purged after 10 days) |

**Clone the repo into `$WORK`:**

```bash
cd $WORK
git clone https://github.com/ashleyscruse/ai-generated-image-detection.git
cd ai-generated-image-detection
```

### TACC Checklist

- [ ] Received TACC invitation email
- [ ] Created account using the invitation link
- [ ] MFA set up and tested
- [ ] Successfully logged in (SSH or Analysis Portal)

---

## 7. Getting Help

- **Stuck for more than 15 minutes?** Message Dr. Scruse. Don't spin your wheels.
- **Git issues?** Screenshot the error and send it.
- **TACC problems?** Check the [TACC documentation](https://docs.tacc.utexas.edu/) or submit a [support ticket](https://portal.tacc.utexas.edu/tacc-consulting).

---

## Quick Reference

| Task | Command |
|------|---------|
| Activate venv | `source venv/bin/activate` |
| Pull updates | `git pull origin main` |
| Switch to your branch | `git checkout your-branch-name` |
| Stage changes | `git add notebooks/portfolios/your_name.ipynb` |
| Commit | `git commit -m "your message"` |
| Push | `git push origin your-branch-name` |
| Check branch | `git branch` |
| SSH to TACC | `ssh user@vista.tacc.utexas.edu` |