#!/bin/bash -l
#SBATCH --job-name=ship_seg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=ship_gpu_%j.log

# --- NUCLEAR CLEANING START ---
# Forcefully unset the variables that cause the Flatpak/Driver errors
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset PYTHONHOME
# --- NUCLEAR CLEANING END ---

# 1. Clean modules
module purge

# 2. Load the optimized System Modules (SCITAS Standard)
# This includes the GPU drivers!
module load gcc python py-tensorflow

# 3. Activate the venv (Assuming you created it as discussed)
# If you didn't create the venv yet, comment this line out and just run on base to test.
source venv/bin/activate

# 4. Verify GPU (Must not be empty!)
echo "Checking GPU visibility..."
python -c "import tensorflow as tf; print('Devices:', tf.config.list_physical_devices('GPU'))"

# 5. Run
echo "Starting training..."
python fast_ynet.py