#!/bin/bash
# Run this ONCE on the cluster after transferring the repo.
# Usage: bash cluster_setup.sh

set -e

CONDA_ENV_NAME="pi3"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Setting up Pi3 on cluster ==="
echo "Repo dir: $REPO_DIR"

# Load conda (adjust module name to match your cluster)
# Try common module names; comment out what doesn't apply
if command -v module &>/dev/null; then
    module load miniconda3 2>/dev/null || \
    module load anaconda3 2>/dev/null || \
    module load conda 2>/dev/null || \
    echo "No conda module found — assuming conda is already in PATH"
fi

# Create conda env if it doesn't exist
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "Conda env '${CONDA_ENV_NAME}' already exists, skipping creation."
else
    echo "Creating conda env '${CONDA_ENV_NAME}' with Python 3.10..."
    conda create -y -n "${CONDA_ENV_NAME}" python=3.10
fi

# Activate and install
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

echo "Installing requirements..."
pip install -r "${REPO_DIR}/requirements.txt"

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate ${CONDA_ENV_NAME}"
echo "Then run:      cd ${REPO_DIR} && python example_mm.py --data_path <your_dataset_path>"
