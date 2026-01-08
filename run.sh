#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8

# 1. Load system modules (Try this first! Adjust versions if needed)
module load cuda/11.8  # or cuda/12.1 depending on your TF version
module load cudnn/8.9

# 2. Activate Conda
source /home/mbouchou/miniconda3/bin/activate shipseg

# 3. CRITICAL: Tell TF where Conda's CUDA libs are (if step 1 doesn't work)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# 4. Debug: Check if GPU is visible before running script
echo "Checking GPU..."
nvidia-smi

python u-net_v2.py