#!/bin/bash
# TACC Setup Script -- run this ONCE after SSHing into Vista
# Usage: ssh ashleyscruse@vista.tacc.utexas.edu
#        bash 01_setup.sh

echo "=== NOBLE Project: TACC Setup ==="

# Clone repo into $WORK
cd $WORK
if [ ! -d "ai-generated-image-detection" ]; then
    git clone https://github.com/ashleyscruse/ai-generated-image-detection.git
    echo "Repo cloned."
else
    cd ai-generated-image-detection && git pull && cd ..
    echo "Repo updated."
fi

cd ai-generated-image-detection

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
fi

source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install kaggle pandas numpy Pillow tqdm pyyaml imagehash

# Set up Kaggle credentials
mkdir -p ~/.kaggle
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo '{"username":"ashleyscruse","key":"KGAT_a9c2c1732e0a5b10e5f8f40c1724a66e"}' > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    echo "Kaggle configured."
fi

# Create data directories
mkdir -p data/raw/videos/ucf_crime
mkdir -p data/raw/videos/police_activity
mkdir -p data/raw/real/surveillance
mkdir -p data/raw/real/bodycam

echo ""
echo "=== Setup complete ==="
echo "Next: sbatch tacc/02_download_ucf.slurm"
