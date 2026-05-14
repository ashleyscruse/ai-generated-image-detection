#!/bin/bash
#SBATCH -J noble_gen_sdxl
#SBATCH -o logs/noble_gen_sdxl_%j.out
#SBATCH -e logs/noble_gen_sdxl_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 16:00:00
#SBATCH -A TRA25001
#SBATCH --mail-user=ashley.scruse@morehouse.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /work2/10539/ashleyscruse/vista/ai-generated-image-detection
module load gcc/14.2.0 cuda/12.6 python3/3.11.8

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "=== SDXL ==="
echo "Started: $(date)"
nvidia-smi || true
python3 -m src.generation.generate --all --model sdxl --num-per-prompt 15 --resolution 1024
echo "Finished: $(date)"
