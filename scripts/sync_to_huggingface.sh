#!/bin/bash
# After frame extraction is done, upload extracted frames to HuggingFace
# Run interactively (not as SLURM job): bash tacc/04_sync_frames_to_hf.sh

echo "=== Sync extracted frames to HuggingFace ==="

cd $WORK/ai-generated-image-detection
source venv/bin/activate

pip install huggingface_hub 2>/dev/null

# Count what we have
SURV_COUNT=$(ls data/raw/real/surveillance/*.jpg 2>/dev/null | wc -l)
BODY_COUNT=$(ls data/raw/real/bodycam/*.jpg 2>/dev/null | wc -l)
OPEN_COUNT=$(find data/raw/real -maxdepth 2 -name "*.jpg" ! -path "*/surveillance/*" ! -path "*/bodycam/*" | wc -l)

echo "Surveillance frames: $SURV_COUNT"
echo "Bodycam frames: $BODY_COUNT"
echo "Open Images: $OPEN_COUNT"
echo "Total real: $((SURV_COUNT + BODY_COUNT + OPEN_COUNT))"
echo ""

python3 -c "
from huggingface_hub import HfApi, login
import os

login(token=os.environ.get('HF_TOKEN', input('Enter HuggingFace token: ')))
api = HfApi()

repo = 'ashleyscruse/noble-ai-evidence-benchmark'

for folder, repo_path in [
    ('data/raw/real/surveillance', 'raw/real/surveillance'),
    ('data/raw/real/bodycam', 'raw/real/bodycam'),
]:
    if os.path.exists(folder) and os.listdir(folder):
        print(f'Uploading {folder}...')
        api.upload_folder(
            folder_path=folder,
            path_in_repo=repo_path,
            repo_id=repo,
            repo_type='dataset',
            ignore_patterns=['.*'],
        )
        print(f'  Done: {repo_path}')

print('Upload complete!')
"

echo "Done: $(date)"
