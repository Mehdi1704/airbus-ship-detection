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
