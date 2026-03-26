#!/bin/bash
#SBATCH --job-name=hac_setup
#SBATCH --output=logs/hac_setup_%j.out
#SBATCH --error=logs/hac_setup_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100-40
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8

set -e
cd "${HOME}"
mkdir -p Pi3/logs

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $(nvcc --version | grep release)"

# ── 1. Clone HAC-plus ─────────────────────────────────────────────────────────
if [ ! -d "HAC-plus" ]; then
    git clone https://github.com/YihangChen-ee/HAC-plus.git
fi
cd HAC-plus

# ── 2. Create conda environment ───────────────────────────────────────────────
eval "$(conda shell.bash hook)"

if conda env list | grep -q "^hac "; then
    echo "Conda env 'hac' already exists, skipping creation."
else
    conda create -y -n hac python=3.12
fi
conda activate hac

# ── 3. Install PyTorch 2.5.1 + CUDA 12.4 ─────────────────────────────────────
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Fix nvidia lib paths (same issue as Pi3)
PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export LD_LIBRARY_PATH=$(find ${CONDA_PREFIX}/lib/python${PYVER}/site-packages/nvidia \
    ~/.local/lib/python${PYVER}/site-packages/nvidia \
    -name "*.so*" -exec dirname {} \; 2>/dev/null | sort -u | tr '\n' ':')${CONDA_PREFIX}/lib

# Verify torch+CUDA works before compiling extensions
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

# ── 4. Install python dependencies ────────────────────────────────────────────
pip install \
    plyfile \
    tqdm \
    einops \
    lpips \
    wandb \
    scipy \
    opencv-python \
    matplotlib

# pytorch-scatter for PyTorch 2.5.1 + CUDA 12.4
pip install torch-scatter \
    -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# ── 5. Unzip and compile submodules ───────────────────────────────────────────
cd submodules

for ZIP in diff-gaussian-rasterization simple-knn gridencoder arithmetic; do
    if [ -f "${ZIP}.zip" ]; then
        echo "Unzipping ${ZIP}..."
        unzip -q -o "${ZIP}.zip"
    else
        echo "WARNING: ${ZIP}.zip not found, skipping."
    fi
done

cd ..

# Compile each submodule (must be done on compute node with GPU)
echo "Compiling diff-gaussian-rasterization..."
pip install submodules/diff-gaussian-rasterization

echo "Compiling simple-knn..."
pip install submodules/simple-knn

echo "Compiling gridencoder..."
pip install submodules/gridencoder

echo "Compiling arithmetic..."
pip install submodules/arithmetic

# ── 6. Save LD_LIBRARY_PATH fix permanently in conda env ─────────────────────
mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
cat > ${CONDA_PREFIX}/etc/conda/activate.d/nvidia_libs.sh << 'EOF'
PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export LD_LIBRARY_PATH=$(find ${CONDA_PREFIX}/lib/python${PYVER}/site-packages/nvidia \
    ~/.local/lib/python${PYVER}/site-packages/nvidia \
    -name "*.so*" -exec dirname {} \; 2>/dev/null | sort -u | tr '\n' ':')${CONDA_PREFIX}/lib
EOF

# ── 7. Verify installation ────────────────────────────────────────────────────
echo ""
echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.version.cuda}')
print(f'GPU     : {torch.cuda.get_device_name(0)}')

from diff_gaussian_rasterization import GaussianRasterizer
print('diff-gaussian-rasterization : OK')

from simple_knn._C import distCUDA2
print('simple-knn                  : OK')

import gridencoder
print('gridencoder                 : OK')

import arithmeticcoding
print('arithmetic                  : OK')
"

echo ""
echo "HAC-plus setup complete."
echo "Activate with: conda activate hac"
echo "NOTE: tmc3 (GPCC codec) still needs to be installed separately for encoding/decoding."
echo "      Training works without it."
