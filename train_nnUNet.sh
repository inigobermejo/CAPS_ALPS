#!/bin/bash -l

#SBATCH -M wice
#SBATCH --account=lp_h_pancreatic_surgery
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=126000MB
#SBATCH --time=08:00:00
#SBATCH --exclude=k28g30
#SBATCH --job-name=CAPS_ALPS_train_nnUnet
#SBATCH --output=inference_ensemble_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=inigo.bermejo@uhasselt.be

set -e

# ==========================================================
# USER SETTINGS (edit these for your own VSC account)
# ==========================================================

# VSC project number
PROJECT="365"

# VSC username
VSC_USER="vsc36501"

# Data directory
DATA_DIR="/data/leuven/${PROJECT}/${VSC_USER}"

source ${DATA_DIR}/miniconda3/etc/profile.d/conda.sh

conda activate nnunet

# ==========================================================
# nnU-Net environment variables
# ==========================================================
# Scratch directory
SCRATCH_DIR="/scratch/leuven/${PROJECT}/${VSC_USER}/CAPS"
# The following directories are required by nnU-Net.
# For inference only, nnUNet_raw and nnUNet_preprocessed may remain empty.
export nnUNet_raw="${SCRATCH_DIR}/nnUNet_raw"
export nnUNet_preprocessed="${SCRATCH_DIR}/nnUNet_preprocessed"
export nnUNet_results="${SCRATCH_DIR}/nnUNet_results"

echo "Node:"
hostname

echo "Starting"
date

echo "Python:"
which python
python --version

echo "nnU-Net:"
which nnUNetv2_train

python ${DATA_DIR}/CAPS/Scripts/ALPS/train_nnUnet.py

echo "Finished"
date
