# MNIST Digit Classifier

**Course:** CT104-3-M Pattern Recognition  
**Institution:** Asia Pacific University of Technology and Innovation (APU)  
**Assignment:** Handwritten Digit Recognition (0-9)

This project builds an end-to-end handwritten digit recognition pipeline using the MNIST dataset in MATLAB.

## Team Members
- LOH HOI PING
- TEE MUN CHUN
- ROHAN MAZUMDAR

## What This Project Does
- Loads raw MNIST CSV data.
- Preprocesses pixel values.
- Extracts multiple feature sets (Raw, PCA, HOG, Hybrid).
- Trains multiple models (k-NN, SVM, Random Forest).
- Evaluates all combinations with clear metrics.
- Includes a simple GUI demo for final presentation.

## Current Project Structure
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
│   └── demo_gui.md
├── data/
│   ├── csv/
│   ├── loaded/
│   ├── preprocessed/
│   └── feature-extracted/
└── results/
    ├── models/
    └── evaluate_model/
```

## Important Git Note
Large generated files are ignored by `.gitignore`.

Ignored examples:
- `*.mat`
- `results/**/*.png`
- `results/**/*.txt`
- `data/csv/*.csv`

## Setup

### 1. Clone
```bash
git clone https://github.com/rohan1443/mnist-digit-classifier.git
cd mnist-digit-classifier
```

### 2. Download Dataset
Download MNIST CSV files from Kaggle:
`https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data`

Place these files in `data/csv/`:
- `mnist_train.csv`
- `mnist_test.csv`

### 3. Verify Files
```bash
ls data/csv/
```

## How To Run

Set MATLAB current directory to the project root first:
```matlab
cd('/<path-to>/mnist-digit-classifier')
```

Then run either:

1. Full interactive menu:
```matlab
run('src/main_pipeline.m')
```

2. Or stage-by-stage:
```matlab
run('src/load_data.m')
run('src/preprocess_data.m')
run('src/extract_features.m')
run('src/train_models.m')
run('src/evaluate_model.m')
```

3. Optional demo GUI (after training + evaluation):
```matlab
demo_gui
```

## Pipeline Flow
1. `load_data.m`
2. `preprocess_data.m`
3. `extract_features.m`
4. `train_models.m`
5. `evaluate_model.m`
6. `demo_gui.m` (presentation/demo)

## Script-by-Script Documentation
Detailed, assignment-friendly explanations are in:

- `docs/README.md`
- `docs/main_pipeline.md`
- `docs/load_data.md`
- `docs/preprocess_data.md`
- `docs/extract_features.md`
- `docs/train_models.md`
- `docs/evaluate_model.md`
- `docs/demo_gui.md`

## Notes For Assignment Report
- Compare at least 3 classifiers and 3+ feature approaches.
- Show validation vs test performance to discuss overfitting.
- Include confusion matrix and per-class metrics (precision/recall/F1).
- Explain why Hybrid features are useful for this task.

## Status
Pipeline implemented and runnable. Documentation updated for final assignment write-up.

