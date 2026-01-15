# Airbus Ship Detection – Semantic Segmentation Project

This repository contains code and models for ship detection in satellite images using deep learning-based semantic segmentation. It leverages a cleaned and preprocessed subset of the [Airbus Ship Detection dataset](https://www.kaggle.com/competitions/airbus-ship-detection/overview).
After preprocessing and cleaning, a subset of **12,788 images** is obtained, which is then cached into local storage for faster lookup when training the model.

---

## Generating the ready-to-use dataset

Download the subset of **12,788 cleaned and preprocessed images** from [Google Drive](https://drive.google.com/drive/folders/1PC9S4bVlWezJVHla-0PTY5gUMTZsLN3N?usp=drive_link) and place them in some folder.  
 
> Note: In case we wanted to start to train the models, we would need to first generate the cached images and masks, by also downloading the cleaned `masks_subset.csv` file from [Google Drive](https://drive.google.com/file/d/1FnoCpy2vjUsD8iZGTghhNfe3XDoseHz8/view?usp=drive_link) (or project folder), updating the 4 path variables at the beginning of the file 'preprocessing/build_cache.py', and running it as:

```bash
python preprocessing/build_cache.py
```

## Using `inference.ipynb` notebook on test set

Note that, currently, this notebook already has some test set image examples saved in the folder `images` within the current directory. However, one can also make it work for any image on the test set by doing the following:

1. Download the test set of **15,606 images** from [Google Drive](https://drive.google.com/drive/folders/1BCn0aBh0GgeQU_HqWbvWlZ-DILSpUW9f?usp=drive_link)
2. In cell 4 within the notebook, set the variable `TEST_DIR` to the path where these images were saved, and uncomment the lines stated in the notebook. 

---

## Repository Structure

### Root Files
- **`README.md`** – Project documentation and setup instructions
- **`requirements.txt`** – Python package dependencies
- **`environment.yaml`** – Python package dependencies
- **`masks_subset.csv`** – CSV file containing ship mask annotations for the cleaned subset
- **`inference.ipynb`** – Jupyter notebook for running inference on test images using trained models

### Directories

#### `preprocessing/`
Contains scripts and notebooks for data preprocessing and exploration:
- **`build_cache.py`** – Generates cached images and masks for faster training
- **`helpers.py`** – Utility functions for preprocessing
- **`dataprocessing.ipynb`** – Data processing exploration and workflow
- **`image_creation.ipynb`** – Image preparation and augmentation
- **`eda-of-the-dataset.ipynb`** – Exploratory Data Analysis
- **`check_images.ipynb`** – Image validation and quality checks
- **`old_explorations/`** – Archive of earlier exploration notebooks

#### `trainings/`
Core training scripts for different model architectures:
- **`FINAL_MODEL.py`** – Final optimized model training script
- **`ResNet34.py`** – ResNet34-based semantic segmentation model
- **`ResNet101.py`** – ResNet101-based semantic segmentation model
- **`UNet.py`** – U-Net based semantic segmentation model
- **`npy_generator.py`** – Data generator for training from numpy arrays

#### `postprocessing/`
Post-training optimization and inference scripts:
- **`postprocessing.py`** – Post-processing pipeline for model predictions
- **`submission_parallel.py`** – Parallel inference for generating submission files
- **`submission_512.py`** – Submission script optimized for 512-pixel images
- **`finetuning.py`** – Fine-tuning script for model improvements
- **`lighter_model.py`** – Model optimization for reduced size/inference time
- **`debug_model.py`** – Debugging utilities for model analysis
- **`debug_model_big.py`** – Extended debugging utilities

#### `models/`
Trained model files:
- **`final_model.tflite`** – Final optimized TensorFlow Lite model
- **`finetuned_model_512.tflite`** – Fine-tuned model for 512-pixel inputs

#### `results/`
Training results and visualizations:
- **`initial_train/`** – Results from initial model training
  - `training_metrics.csv` – Initial training metrics
- **`finetuning/`** – Results from fine-tuning experiments
  - `training_metrics.csv` – Fine-tuning metrics
- **`visual_debug_50/`** – Visual debugging outputs (50 samples)
- **`visual_debug_old_50/`** – Archive of older visual debugging outputs

#### `final_submissions/`
Final submission files for competitions/evaluations:
- **`resnet34.csv`** – ResNet34 predictions
- **`resnet34FN.csv`** – ResNet34 with false negative handling
- **`resnet34_FN+filtered.csv`** – ResNet34 with filtered false negatives
- **`resnet101.csv`** – ResNet101 predictions
- **`unet.csv`** – U-Net predictions

#### `cluster/`
Scripts for distributed/cluster computing:
- **`run.sh`** – Main cluster execution script
- **`run2.sh`** – Alternative cluster execution variant
- **`run3.sh`** – Another cluster execution variant

#### `logs/`
Logging utilities:
- **`log_processing.py`** – Script for processing and analyzing logs

#### `images/`
Test images for inference demonstrations

---

