#!/bin/bash
#SBATCH -J noble_eval_hf
#SBATCH -o logs/noble_eval_hf_%j.out
#SBATCH -e logs/noble_eval_hf_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TRA25001
#SBATCH --mail-user=ashley.scruse@morehouse.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /work2/10539/ashleyscruse/vista/ai-generated-image-detection
module load gcc/14.2.0 cuda/12.6 python3/3.11.8

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "=== Evaluation: HuggingFace detector vs full benchmark ==="
echo "Started: $(date)"
nvidia-smi || true

# Run for each degradation level (script supports --level OR runs all if omitted)
for level in clean moderate heavy; do
    echo ""
    echo ">>> Level: $level"
    python3 -m src.evaluation.evaluate --tool huggingface --level $level
done

echo ""
echo "=== Generating figures and summary ==="
python3 -m src.evaluation.evaluate --report

echo ""
echo "Finished: $(date)"
