#!/bin/bash
#SBATCH -J noble_bodycam_download
#SBATCH -o noble_bodycam_download_%j.out
#SBATCH -e noble_bodycam_download_%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A TRA25001
#SBATCH --mail-user=ashley.scruse@morehouse.edu
#SBATCH --mail-type=END,FAIL

# Download PoliceActivity bodycam archive from Internet Archive
# and extract frames on TACC
# Submit: sbatch tacc/03_download_policeactivity.slurm

echo "=== NOBLE: PoliceActivity Bodycam Download + Frame Extraction ==="
echo "Started: $(date)"
echo "Node: $(hostname)"

cd $WORK2/vista/ai-generated-image-detection
source venv/bin/activate

VIDEO_DIR="data/raw/videos/police_activity"
FRAME_DIR="data/raw/real/bodycam"

# Load ffmpeg
module load ffmpeg 2>/dev/null || echo "Using system ffmpeg"

mkdir -p $VIDEO_DIR
mkdir -p $FRAME_DIR

# Step 1: Download from Internet Archive
# The PoliceActivity collection has 944 videos (~44 GB total)
# We'll download a manageable subset
echo ""
echo "--- Step 1: Downloading PoliceActivity videos from Internet Archive ---"

if [ ! -f "$VIDEO_DIR/.download_complete" ]; then
    # Install ia CLI tool
    pip install internetarchive 2>/dev/null

    # Download the collection (MP4 files only, limit to keep manageable)
    ia download policeactivity-20210727 \
        --glob="*.mp4" \
        --destdir=$VIDEO_DIR \
        --no-directories \
        2>&1 | tail -5

    # If ia tool fails, fall back to wget for a subset
    if [ $? -ne 0 ]; then
        echo "ia tool failed, trying wget for metadata..."
        # Get the file listing
        wget -q "https://archive.org/download/policeactivity-20210727/" \
            -O $VIDEO_DIR/listing.html 2>/dev/null

        # Download first 200 MP4 files directly
        wget -q "https://archive.org/download/policeactivity-20210727/" \
            -A "*.mp4" \
            -P $VIDEO_DIR \
            --no-directories \
            --limit-rate=10M \
            -w 1 \
            2>&1 | tail -5
    fi

    touch $VIDEO_DIR/.download_complete
    echo "Download complete."
else
    echo "Already downloaded, skipping."
fi

VIDEO_COUNT=$(find $VIDEO_DIR -name "*.mp4" | wc -l)
echo "Videos available: $VIDEO_COUNT"

# Step 2: Extract frames
echo ""
echo "--- Step 2: Extracting frames ---"

TOTAL_FRAMES=0
MAX_TOTAL=5000

find $VIDEO_DIR -name "*.mp4" -type f | sort | head -500 | while read VIDEO; do
    if [ $TOTAL_FRAMES -ge $MAX_TOTAL ]; then
        echo "Reached $MAX_TOTAL frames, stopping."
        break
    fi

    BASENAME=$(basename "$VIDEO" | sed 's/\.[^.]*$//')

    # Skip if frames already exist
    EXISTING=$(ls $FRAME_DIR/${BASENAME}_*.jpg 2>/dev/null | wc -l)
    if [ $EXISTING -ge 5 ]; then
        continue
    fi

    ffmpeg -i "$VIDEO" \
        -vf "fps=0.1" \
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
echo "Total bodycam frames extracted: $FINAL_COUNT"
echo "Output directory: $FRAME_DIR"
echo "Finished: $(date)"
