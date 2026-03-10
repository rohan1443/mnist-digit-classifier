# `load_data.m`

## What This Script Is
This is the entry point for data preparation. It reads MNIST CSV files and creates proper train/validation/test splits.

## What Is Happening Inside
- Reads:
  - `data/csv/mnist_train.csv`
  - `data/csv/mnist_test.csv`
- Separates label column from pixel columns.
- Splits original training set into:
  - training (80%)
  - validation (20%)
- Keeps test set separate for final unbiased evaluation.
- Saves prepared arrays into:
  - `data/loaded/mnist_data.mat`

## Approach Used
The script uses `cvpartition(..., 'HoldOut', 0.2)` for random holdout split.

Why this is good:
- validation data is not used for fitting model weights
- helps tune hyperparameters fairly
- keeps test data untouched until final stage

## Data Shape in Plain Words
- Each image is 28x28 = 784 values.
- CSV row format:
  - first value = digit label (0-9)
  - next 784 values = grayscale pixel values

## Input and Output
Input:
- `mnist_train.csv`, `mnist_test.csv`

Output:
- `train_images`, `train_labels`
- `val_images`, `val_labels`
- `test_images`, `test_labels`
- saved in `mnist_data.mat`

## Formula Ideas Used
No ML formula here yet, but split logic is critical.

Validation size:
`N_val = 0.2 * N_train`

Training size:
`N_train_new = 0.8 * N_train`
