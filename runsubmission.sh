#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8

# --- 1. Load the Exact Module Found ---
# We use 12.1.1 as it matches your newer driver better.
# If this fails later, try changing it to cuda/11.8.0
module load cuda/12.1.1

# --- 2. Activate Environment ---
source /home/mbouchou/miniconda3/bin/activate shipseg

# --- 3. Link Conda Libraries (The "Safety Net") ---
# This ensures TF finds the libraries inside Conda if the module above isn't enough.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/

# --- 5. Run Python ---
# python -u ensures logs are printed instantly to the output file
python -u visualization.py