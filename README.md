# Airbus Ship Detection – Semantic Segmentation Project

This repository contains code and models for ship detection in satellite images using deep learning-based semantic segmentation. It leverages a cleaned and preprocessed subset of the [Airbus Ship Detection dataset](https://www.kaggle.com/competitions/airbus-ship-detection/overview).
After preprocessing and cleaning, a subset of **12,788 images** is obtained, which is then cached into local storage for faster lookup when training the model.

---

## Generating the ready-to-use dataset

1. Download the subset of **12,788 cleaned and preprocessed images** from [Google Drive](https://drive.google.com/drive/folders/1PC9S4bVlWezJVHla-0PTY5gUMTZsLN3N?usp=drive_link) and place them in a folder (e.g., `/images`).  
2. Make sure the CSV file `masks_subset.csv` is present in the project folder.  
3. Run the cache building script:

```bash
python build_cache.py
