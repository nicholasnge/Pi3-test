#!/bin/bash
#SBATCH --job-name=pi3_colmap
#SBATCH --output=logs/pi3_colmap_%A_%a.out
#SBATCH --error=logs/pi3_colmap_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100-40
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-10

SCENES=(bicycle bonsai Caterpillar Family Francis garden Horse Ignatius kitchen Train Truck)
SCENE="${SCENES[$SLURM_ARRAY_TASK_ID]}"
DATASET_ROOT="${HOME}/3DGSDATASETS"
IMG_DIR="${DATASET_ROOT}/${SCENE}/images"
OUT_DIR="${DATASET_ROOT}/${SCENE}/sparse_pi3/0"
CKPT="${HOME}/Pi3/ckpts/model.safetensors"

set -e
cd "${HOME}/Pi3"
mkdir -p logs "${OUT_DIR}"

eval "$(conda shell.bash hook)"
conda activate pi3

unset LD_LIBRARY_PATH
PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export LD_LIBRARY_PATH=$(find ${CONDA_PREFIX}/lib/python${PYVER}/site-packages/nvidia \
    ~/.local/lib/python${PYVER}/site-packages/nvidia \
    -name "*.so*" -exec dirname {} \; 2>/dev/null | sort -u | tr '\n' ':')${CONDA_PREFIX}/lib

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Scene: ${SCENE}"
echo "Images: ${IMG_DIR}"
echo "Output: ${OUT_DIR}"

python pi3_to_colmap.py \
    --images  "${IMG_DIR}" \
    --output  "${OUT_DIR}" \
    --ckpt    "${CKPT}" \
    --chunk   32 \
    --overlap 8 \
    --conf_thr 0.1 \
    --max_pts 200000 \
    --interval 1

echo "Done: ${SCENE} -> ${OUT_DIR}"
