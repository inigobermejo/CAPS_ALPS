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
#SBATCH --job-name=CAPS_ALPS_predict_nnUnet
#SBATCH --output=log/CAPS_ALPS_nnUNet_predict_cropped_from_finetuned_%j.out
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
export nnUNet_raw="${SCRATCH_DIR}/	"
export nnUNet_preprocessed="${SCRATCH_DIR}/nnUNet_preprocessed"
export nnUNet_results="${SCRATCH_DIR}/nnUNet_results"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

export nnUNet_n_proc_DA=4
export nnUNet_compile=F

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ------------------------------------------------------------
# Environment diagnostics
# ------------------------------------------------------------

echo "============================================================"
echo "JOB START"
echo "============================================================"

echo "Node:"
hostname

echo "Starting"
date

echo "Python:"
which python
python --version

date
hostname

echo "GPU:"
nvidia-smi

echo "CPU:"
nproc

echo "Memory:"
free -h

echo "Disk:"
df -h "$nnUNet_preprocessed"
df -h "$nnUNet_results"

echo "Environment:"
env | grep -E 'nnUNet|CUDA|TORCH|TRITON|TMP|OMP|MKL'

echo "============================================================"
echo "STARTING PREDICTING"
echo "============================================================"

nnUNetv2_predict \
    -i "${DATA_DIR}/CAPS/Jessa/nnUNet/nnUNet_raw/Dataset001_PancreasTumor/imagesTr_fold0" \
    -o "${DATA_DIR}/CAPS/Jessa/nnUNet/nnUNet_raw/Dataset001_PancreasTumor/predictions/fold0" \
    -d 1 \
    -c 3d_fullres \
    -p nnUNetResEncUNetLPlans \
    -f 0 \
    -chk checkpoint_best.pth
echo "Finished"
date
