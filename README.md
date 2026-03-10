# MNIST Digit Classifier

**Course:** CT104-3-M Pattern Recognition  
**Institution:** Asia Pacific University of Technology and Innovation (APU)  
**Assignment:** Handwritten Digit Recognition (0-9)

This repository contains a complete MATLAB-based handwritten digit recognition system using MNIST CSV data. It includes the full machine learning pipeline, evaluation workflow, and a presentation-ready GUI for inference.

## Team Members
- LOH HOI PING
- TEE MUN CHUN
- ROHAN MAZUMDAR

## Quick Glimpse
The screenshots below give a quick view of what was built.

![GUI Initial Screen](docs/assets/run_demo_step_1.png)
![GUI Final Prediction](docs/assets/run_demo_step_3.png)

## Table of Contents
- [Project Scope](#project-scope)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Approach and Design Rationale](#approach-and-design-rationale)
- [How to Run](#how-to-run)
- [Outputs and Artifacts](#outputs-and-artifacts)
- [Documentation Map](#documentation-map)
- [Engineering and Repository Standards](#engineering-and-repository-standards)
- [Status](#status)

## Project Scope
The project is designed to compare multiple feature extraction and classification strategies on the same dataset and evaluation protocol.

Main capabilities:
- Load and split MNIST data into train, validation, and test sets.
- Normalize and prepare data for stable training.
- Extract multiple feature representations.
- Train multiple classifiers with comparable settings.
- Evaluate all combinations using consistent metrics.
- Demonstrate real-time inference in a GUI.

## System Architecture
```text
MNIST CSV (train/test)
                |
                v
load_data.m
    - parse labels/pixels
    - train/validation split
                |
                v
preprocess_data.m
    - normalization
    - data quality checks
                |
                v
extract_features.m
    - Raw, PCA50, PCA100, HOG, Hybrid
                |
                v
train_models.m
    - k-NN, SVM, Random Forest
    - 15 model-feature combinations
                |
                v
evaluate_model.m
    - accuracy, confusion matrix
    - precision/recall/F1/specificity
                |
                v
demo_gui.m
    - image upload
    - prediction + confidence + top-3 probabilities
```

## Repository Structure
```text
mnist-digit-classifier/
├── README.md
├── .gitignore
├── src/
│   ├── main_pipeline.m
│   ├── load_data.m
│   ├── preprocess_data.m
│   ├── extract_features.m
│   ├── train_models.m
│   ├── evaluate_model.m
│   └── demo_gui.m
├── docs/
│   ├── README.md
│   ├── main_pipeline.md
│   ├── load_data.md
│   ├── preprocess_data.md
│   ├── extract_features.md
│   ├── train_models.md
│   ├── evaluate_model.md
│   ├── demo_gui.md
│   ├── demo_walkthrough_gui.md
│   └── assets/
├── data/
│   ├── csv/
│   ├── loaded/
│   ├── preprocessed/
│   └── feature-extracted/
└── results/
        ├── models/
        └── evaluate_model/
```

## Approach and Design Rationale

### Data Handling
- MNIST CSV is used for transparent preprocessing and reproducibility.
- A validation split is used to tune model settings before test evaluation.

### Preprocessing
- Pixel normalization to `[0, 1]` stabilizes model behavior.
- Data checks are included to catch invalid values.

### Feature Engineering
Five feature representations are compared:
- Raw Pixels: baseline reference.
- PCA50 and PCA100: reduced dimensions for speed and compression.
- HOG: shape/edge-focused representation suitable for digits.
- Hybrid (HOG + PCA): combines discriminative shape features with compact vectors.

### Model Selection
Three model families are used for balanced comparison:
- k-NN: simple distance-based baseline.
- SVM: strong margin-based classifier for high-dimensional spaces.
- Random Forest: robust ensemble baseline.

Total configurations:
- `5 feature sets x 3 model families = 15 experiments`

### Evaluation Strategy
- Use the held-out test set only for final performance reporting.
- Report both aggregate and per-class behavior:
    - accuracy
    - confusion matrices
    - precision, recall, F1-score
    - sensitivity and specificity

## How to Run

### Prerequisites
- MATLAB installed
- MNIST CSV files downloaded from:
    - `https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data`

### Dataset Placement
Place files in `data/csv/`:
- `mnist_train.csv`
- `mnist_test.csv`

### Start from Project Root
```matlab
cd('/<path-to>/mnist-digit-classifier')
```

### Run Modes

1. Interactive pipeline menu:
```matlab
run('src/main_pipeline.m')
```

2. Stage-by-stage execution:
```matlab
run('src/load_data.m')
run('src/preprocess_data.m')
run('src/extract_features.m')
run('src/train_models.m')
run('src/evaluate_model.m')
```

3. GUI inference demo (after training + evaluation):
```matlab
run('src/demo_gui.m')
```

## Outputs and Artifacts

Primary outputs:
- `data/loaded/mnist_data.mat`
- `data/preprocessed/mnist_preprocessed.mat`
- `data/feature-extracted/features.mat`
- `results/training_results.mat`
- `results/evaluate_model/evaluation_metrics.mat`
- `results/evaluate_model/final_report.txt`
- `results/evaluate_model/confusion_matrices/*.png`

## Documentation Map
Detailed documentation is organized for fast navigation:

- `docs/README.md`
- `docs/main_pipeline.md`
- `docs/load_data.md`
- `docs/preprocess_data.md`
- `docs/extract_features.md`
- `docs/train_models.md`
- `docs/evaluate_model.md`
- `docs/demo_gui.md`
- `docs/demo_walkthrough_gui.md`

## Engineering and Repository Standards
- Generated artifacts are excluded using `.gitignore`.
- Source scripts and documentation are versioned.
- Stage outputs are separated by folder to keep the pipeline modular.
- Documentation is linked and organized for maintainability and review.

Ignored examples:
- `*.mat`
- `results/**/*.png`
- `results/**/*.txt`
- `data/csv/*.csv`

## Status
Pipeline is implemented and runnable. Documentation is structured for assignment submission, review, and demonstration.

