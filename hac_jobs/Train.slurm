#!/bin/bash
#SBATCH --job-name=hac_Train
#SBATCH --output=/home/n/nicnge21/HAC/logs/Train_%j.out
#SBATCH --error=/home/n/nicnge21/HAC/logs/Train_%j.err
#SBATCH --gres=gpu:a100-40
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8

eval "$(conda shell.bash hook)"
conda activate HAC_env

mkdir -p /home/n/nicnge21/HAC/logs
cd /home/n/nicnge21/HAC

CUDA_VISIBLE_DEVICES=0 python train.py \
    -s data/pi3/Train \
    -m outputs/pi3/Train/0.004 \
    --eval \
    --lod 0 \
    --voxel_size 0.01 \
    --update_init_factor 16 \
    --iterations 30000 \
    --lmbda 0.004 \
    --mask_lr_final 0.0004
