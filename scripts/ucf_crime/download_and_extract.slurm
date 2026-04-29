#!/bin/bash
#SBATCH -J noble_ucf_download
#SBATCH -o noble_ucf_download_%j.out
#SBATCH -e noble_ucf_download_%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A TRA25001
#SBATCH --mail-user=ascruse@morehouse.edu
#SBATCH --mail-type=END,FAIL

# Download UCF Crime Dataset and extract surveillance frames on TACC
# Submit: sbatch tacc/02_download_ucf.slurm

echo "=== NOBLE: UCF Crime Download + Frame Extraction ==="
echo "Started: $(date)"
echo "Node: $(hostname)"

cd $WORK2/vista/ai-generated-image-detection
source venv/bin/activate

VIDEO_DIR="data/raw/videos/ucf_crime"
FRAME_DIR="data/raw/real/surveillance"

# Step 1: Download from Kaggle
echo ""
echo "--- Step 1: Downloading UCF Crime Dataset (~13 GB) ---"
if [ ! -f "$VIDEO_DIR/.download_complete" ]; then
    kaggle datasets download -d odins0n/ucf-crime-dataset -p $VIDEO_DIR --unzip
    touch $VIDEO_DIR/.download_complete
    echo "Download complete."
else
    echo "Already downloaded, skipping."
fi

# Step 2: Extract frames
echo ""
echo "--- Step 2: Extracting frames ---"

# Load ffmpeg module (TACC provides this)
module load ffmpeg 2>/dev/null || echo "ffmpeg module not found, using system ffmpeg"

mkdir -p $FRAME_DIR

# Extract 1 frame every 5 seconds, max 10 per video
TOTAL_FRAMES=0
MAX_TOTAL=5000

find $VIDEO_DIR -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mpg" \) | sort | while read VIDEO; do
    if [ $TOTAL_FRAMES -ge $MAX_TOTAL ]; then
        echo "Reached $MAX_TOTAL frames, stopping."
        break
    fi

    BASENAME=$(basename "$VIDEO" | sed 's/\.[^.]*$//')

    # Skip if frames already exist for this video
    EXISTING=$(ls $FRAME_DIR/${BASENAME}_*.jpg 2>/dev/null | wc -l)
    if [ $EXISTING -ge 5 ]; then
        continue
    fi

    ffmpeg -i "$VIDEO" \
        -vf "fps=0.2" \
        -frames:v 10 \
        -q:v 2 \
        "$FRAME_DIR/${BASENAME}_%04d.jpg" \
        -y -loglevel error 2>/dev/null

    NEW_FRAMES=$(ls $FRAME_DIR/${BASENAME}_*.jpg 2>/dev/null | wc -l)
    TOTAL_FRAMES=$((TOTAL_FRAMES + NEW_FRAMES))

    if [ $((TOTAL_FRAMES % 100)) -eq 0 ]; then
        echo "  Extracted $TOTAL_FRAMES frames so far..."
    fi
done

FINAL_COUNT=$(ls $FRAME_DIR/*.jpg 2>/dev/null | wc -l)
echo ""
echo "--- Results ---"
echo "Total frames extracted: $FINAL_COUNT"
echo "Output directory: $FRAME_DIR"
echo "Finished: $(date)"
