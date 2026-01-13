#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=32


python submission_parallel.py