#!/bin/bash
#SBATCH -J noble_gen_rv51
#SBATCH -o logs/noble_gen_rv51_%j.out
#SBATCH -e logs/noble_gen_rv51_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -A TRA25001
#SBATCH --mail-user=ashley.scruse@morehouse.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /work2/10539/ashleyscruse/vista/ai-generated-image-detection
module load gcc/14.2.0 cuda/12.6 python3/3.11.8

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "=== Realistic Vision 5.1 ==="
echo "Started: $(date)"
nvidia-smi || true
python3 -m src.generation.generate --all --model rv51 --num-per-prompt 15 --resolution 512
echo "Finished: $(date)"
