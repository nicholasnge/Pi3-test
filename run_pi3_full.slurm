#!/bin/bash
#SBATCH --job-name=pi3_full
#SBATCH --output=logs/pi3_full_%j.out
#SBATCH --error=logs/pi3_full_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100-40
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4

SCENES=(bicycle bonsai Caterpillar Family Francis garden Horse Ignatius kitchen Train Truck)
DATASET_ROOT="${HOME}/3DGSDATASETS"

set -e
cd "${HOME}/Pi3"
mkdir -p logs

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

for SCENE in "${SCENES[@]}"; do
    echo "========================================="
    echo "Starting: ${SCENE}"
    echo "========================================="
    python pi3_run.py \
        --scene_dir "${DATASET_ROOT}/${SCENE}" \
        --out_dir   "outputs/${SCENE}" \
        --ckpt      "${HOME}/Pi3/ckpts/model.safetensors" \
        --interval  1 \
        --max_pts   200000 \
        --chunk     32 \
        --overlap   8 \
        --probe_n   12
    echo "Done: ${SCENE}"
done

echo "All scenes complete."
