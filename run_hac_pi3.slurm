#!/bin/bash
#SBATCH --job-name=hac_pi3
#SBATCH --output=logs/hac_pi3_%j.log
#SBATCH --error=logs/hac_pi3_%j.log
#SBATCH --gres=gpu:a100-40
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8

set -e
cd "${HOME}/HAC"
mkdir -p "${HOME}/Pi3/logs"

eval "$(conda shell.bash hook)"
conda activate HAC_env

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Scene configs ─────────────────────────────────────────────────────────────
# mip360: voxel_size=0.001  |  tnt: voxel_size=0.01
# mask_lr_final differs per dataset (from original HAC-plus scripts)

DATA_ROOT="data/pi3"

run_scene() {
    local SCENE=$1
    local VOXEL=$2
    local MASK_LR=$3
    local LMBDA=$4

    echo "========================================="
    echo "Scene: ${SCENE} | lmbda: ${LMBDA} | voxel: ${VOXEL}"
    echo "========================================="

    CUDA_VISIBLE_DEVICES=0 python train.py \
        -s "${DATA_ROOT}/${SCENE}" \
        -m "outputs/pi3/${SCENE}/${LMBDA}" \
        --eval \
        --lod 0 \
        --voxel_size ${VOXEL} \
        --update_init_factor 16 \
        --iterations 30000 \
        --lmbda ${LMBDA} \
        --mask_lr_final ${MASK_LR}
}

# Lambda to use — add more values to the array to sweep
for LMBDA in 0.004; do

    # mask_lr_final formulas from original HAC-plus
    MIP_MASK_LR=$(python -c "print(min(0.0005 * ${LMBDA} / 0.001, 0.0015))")
    TNT_MASK_LR=$(python -c "print(0.0001 * ${LMBDA} / 0.001)")

    # ── Mip-NeRF 360 scenes ───────────────────────────────────────────────────
    for SCENE in bicycle bonsai garden kitchen; do
        run_scene "${SCENE}" 0.001 "${MIP_MASK_LR}" "${LMBDA}"
    done

    # ── Tanks & Temples scenes ────────────────────────────────────────────────
    for SCENE in Truck Train Caterpillar Family Francis Horse Ignatius; do
        run_scene "${SCENE}" 0.01 "${TNT_MASK_LR}" "${LMBDA}"
    done

done

echo "All HAC-plus Pi3 training complete."
