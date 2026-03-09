# MNIST Digit Classifier – Pattern Recognition Project

**Team:** LOH HOI PING, TEE MUN CHUN, ROHAN MAZUMDAR  
**Course:** CT104-3-M Pattern Recognition  
**Institution:** Asia Pacific University (APU)

## Overview
This project implements a complete machine learning pipeline for handwritten digit recognition using the MNIST dataset. It compares multiple classifiers (Naive Bayes, LDA, Logistic Regression, SVM) and feature extraction methods (raw pixels, HOG, PCA).

## Requirements
- MATLAB (R2020b or later) with Statistics and Machine Learning Toolbox
- MNIST CSV files (download from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)) placed in `data/csv/`

## How to Run
1. Clone this repository.
2. Place `mnist_train.csv` and `mnist_test.csv` in `data/csv/`.
3. Open MATLAB and navigate to the project root.
4. Run `main_pipeline.m` – it will execute all stages sequentially.

## Pipeline Stages
- **Data Loading** (`load_data.m`) – loads CSV, creates train/validation split.
- **Preprocessing** (`preprocess_data.m`) – normalisation, mean centering, per‑pixel standardisation.
- **Feature Extraction** (`extract_features.m`) – raw pixels, HOG, PCA.
- **Training** – separate scripts for each classifier (LDA, logistic regression, Naive Bayes, SVM).
- **Evaluation** – computes test accuracy, confusion matrices, and logs.

## Results
All outputs are saved in the `results/` folder:
- `*_log.txt` – detailed logs of each stage.
- `**/*.png` – confusion matrices for each classifier.

Trained models are saved in `models/<classifier>/`.

## Key Findings
- HOG features dramatically improve accuracy for all classifiers.
- Linear SVM on HOG achieved the highest accuracy (98.69%).
- Naive Bayes underperformed due to its independence assumption.

## References
[1] LeCun et al. (1998). Gradient-based learning applied to document recognition.  
[2] Cortes & Vapnik (1995). Support-vector networks.  
[3] Dalal & Triggs (2005). Histograms of oriented gradients for human detection.