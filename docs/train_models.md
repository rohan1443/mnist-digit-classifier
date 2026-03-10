# `train_models.m`

## What This Script Is
This stage trains classifiers using every feature set and records validation performance.

## Models Trained
1. k-NN
2. SVM (ECOC one-vs-all for multiclass)
3. Random Forest (TreeBagger)

## Total Experiments
- 5 feature sets x 3 models = 15 combinations

## Process
- Load all extracted feature sets.
- Loop over each feature set.
- Train model, validate, tune key hyperparameters.
- Save each trained model under `results/models/`.
- Store validation results summary in `results/training_results.mat`.

## Hyperparameters Tuned
k-NN:
- `k` from `[3, 5, 7, 10]`

SVM:
- Linear kernel with fixed box constraint in current implementation

Random Forest:
- number of trees from `[50, 100]`

## Easy Formula View
Validation accuracy:
`accuracy = correct_predictions / total_validation_samples`

For k-NN:
- prediction comes from majority label among nearest `k` points.

For Random Forest:
- prediction is voting result across decision trees.

## Why This Stage Is Important
- Gives fair comparison across methods.
- Produces actual evidence for "best model" claim.
- Saves reproducible model files for final evaluation.

## Output Files
- `results/models/knn_*.mat`
- `results/models/svm_*.mat`
- `results/models/rf_*.mat`
- `results/training_results.mat`
