#!/bin/bash -l
#SBATCH --job-name=ship_seg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --time=02:00:00
#SBATCH --output slurm-%J.out

# --- THE FIX IS HERE ---
# 1. Purge any old modules that might conflict
module purge

# 2. Load the official SCITAS TensorFlow module (includes CUDA drivers)
module load gcc python openmpi py-tensorflow

# 3. Run your script directly (do NOT use flatpak or apptainer unless configured with --nv)
python fast_ynet.py