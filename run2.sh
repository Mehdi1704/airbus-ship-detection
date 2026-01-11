#!/bin/bash
#SBATCH --job-name=ship_seg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=ship_training_%j.log

# 1. Activate Conda
source ~/.bashrc
conda activate shipseg

# 2. CRITICAL: Force TensorFlow to use Conda's GPU libraries
# This bypasses the need for "module load cuda"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

# 3. Debug: Verify the GPU is visible before starting
echo "Checking GPU visibility..."
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 4. Run the Fast Script
echo "Starting training..."
python fast_ynet.py