#!/bin/bash
# setup_hac_data.sh
# Creates the data directory structure HAC expects, using Pi3 sparse outputs
# and symlinking original images.
# Run this AFTER pi3_run.py has completed for all scenes.
#
# Usage: bash setup_hac_data.sh

set -e

PI3_OUTPUTS="${HOME}/Pi3/outputs"
DATASETS="${HOME}/3DGSDATASETS"
HAC_DATA="${HOME}/HAC/data/pi3"

mkdir -p "${HAC_DATA}"

MIP360=(bicycle bonsai garden kitchen)
TNT=(Caterpillar Family Francis Horse Ignatius Train Truck)

for SCENE in "${MIP360[@]}" "${TNT[@]}"; do
    echo "Setting up: ${SCENE}"
    SCENE_DIR="${HAC_DATA}/${SCENE}"
    mkdir -p "${SCENE_DIR}"

    # Symlink images from original dataset
    if [ ! -e "${SCENE_DIR}/images" ]; then
        ln -s "${DATASETS}/${SCENE}/images" "${SCENE_DIR}/images"
    fi

    # Symlink sparse/0 from Pi3 outputs
    mkdir -p "${SCENE_DIR}/sparse"
    if [ ! -e "${SCENE_DIR}/sparse/0" ]; then
        ln -s "${PI3_OUTPUTS}/${SCENE}/sparse/0" "${SCENE_DIR}/sparse/0"
    fi

    # Verify
    if [ -f "${SCENE_DIR}/sparse/0/cameras.txt" ] && \
       [ -f "${SCENE_DIR}/sparse/0/images.txt" ] && \
       [ -f "${SCENE_DIR}/sparse/0/points3D.txt" ]; then
        echo "  OK: sparse/0/ complete"
    else
        echo "  WARNING: sparse/0/ missing files — has pi3_run.py finished for ${SCENE}?"
    fi
done

echo ""
echo "Data structure ready at ${HAC_DATA}"
echo "Run: sbatch ~/Pi3/run_hac_pi3.slurm"
