# `main_pipeline.m` Interactive Run Walkthrough

## Table of Contents
- [What This Script Is](#what-this-script-is)
- [What It Does](#what-it-does)
- [Stage Order Used by the Pipeline](#stage-order-used-by-the-pipeline)
- [Option 3: View Existing Results](#option-3-view-existing-results)
- [Option 2 -> 1: Run Individual Step (Load Data)](#option-2---1-run-individual-step-load-data)
- [Option 1: Run Complete Pipeline](#option-1-run-complete-pipeline)
- [Snapshot Placeholders (Add After Sharing)](#snapshot-placeholders-add-after-sharing)
- [Why This Matters For the Assignment](#why-this-matters-for-the-assignment)

## What This Script Is
This is the master launcher of the full project. It is the control center that lets you run the entire workflow or specific parts based on your needs.

## What It Does
- Shows a simple menu-based interface.
- Lets user run:
  - complete pipeline
  - one individual step
  - existing results summary
- Coordinates stage execution in the correct order.

## Stage Order Used by the Pipeline
1. `load_data.m`
2. `preprocess_data.m`
3. `extract_features.m`
4. `train_models.m`
5. `evaluate_model.m`

## Option 3: View Existing Results
For results-view mode, run and choose:

```text
Enter choice (0-3): 3
```

This mode loads saved evaluation metrics and displays:
- best model and feature set
- accuracy comparison table across all combinations
- detailed per-digit metrics for the best model

### Console Flow (Example Output)

```text
=============================================================
   MNIST HANDWRITTEN DIGIT RECOGNITION SYSTEM
   Pattern Recognition Assignment - CT104-3-M
=============================================================

Select operation:
  1 - Run complete pipeline (all steps)
  2 - Run individual step
  3 - View existing results
  0 - Exit

Enter choice (0-3): 3

=== View Results ===

=== RESULTS SUMMARY ===

BEST MODEL:
  Algorithm: SVM
  Features: HOG
  Test Accuracy: 98.69%

TEST ACCURACIES:

Feature Set     | k-NN       | SVM        | Random Forest
----------------+------------+------------+-----------------
Raw             |     96.98% |     92.06% |          96.78%
PCA50           |     91.63% |     58.35% |          74.26%
PCA100          |     92.01% |     57.94% |          76.76%
HOG             |     97.98% |     98.69% |          98.23%
Hybrid          |     95.68% |     75.83% |          85.71%

DETAILED METRICS (Best Model):

Digit | Precision | Recall | F1-Score
------+-----------+--------+---------
  0   |   98.89%   | 99.59% | 99.24%
  1   |   99.04%   | 99.47% | 99.25%
  2   |   98.56%   | 99.22% | 98.89%
  3   |   98.62%   | 98.81% | 98.71%
  4   |   98.57%   | 98.57% | 98.57%
  5   |   98.77%   | 98.99% | 98.88%
  6   |   99.27%   | 98.75% | 99.01%
  7   |   98.34%   | 97.76% | 98.05%
  8   |   98.56%   | 98.15% | 98.35%
  9   |   98.30%   | 97.52% | 97.91%

Full report available in: results/final_report.txt
```

## Option 2 -> 1: Run Individual Step (Load Data)
This path runs only the load-data stage, useful for quick verification.

```text
=============================================================
   MNIST HANDWRITTEN DIGIT RECOGNITION SYSTEM
   Pattern Recognition Assignment - CT104-3-M
=============================================================

Select operation:
  1 - Run complete pipeline (all steps)
  2 - Run individual step
  3 - View existing results
  0 - Exit

Enter choice (0-3): 2

=== Run Individual Step ===

Select step to run:
  1 - Load Data
  2 - Preprocess Data
  3 - Extract Features
  4 - Train Models
  5 - Evaluate Models
  0 - Back

Enter step (0-5): 1

Loading MNIST dataset...
Files loaded successfully!

Training set size: 48000
Validation set size: 12000
Test set size: 10000
Data saved to mnist_data.mat successfully!
```

### What This Confirms
- Dataset files are found correctly.
- Split sizes are created as expected.
- Output for next step is generated:
  - `data/loaded/mnist_data.mat`

## Option 1: Run Complete Pipeline
When you choose `1`, all stages run in sequence from loading data to evaluation.

Because this mode is long-running (especially during feature extraction and model training), full raw terminal output is not practical to paste in documentation. A concise report-style summary is preferred:
- pipeline started and ran stage-by-stage
- each major stage completed successfully
- model files and metrics were saved
- final outputs generated under `results/` and `results/evaluate_model/`

## Snapshot Placeholders (Add After Sharing)
Once snapshots are shared, place them in `docs/assets/` and embed them here:

```markdown
![Main Pipeline Menu](docs/assets/main_pipeline_view_results_step_1.png)
![Results Summary Output](docs/assets/main_pipeline_view_results_step_2.png)
![Per-Digit Metrics Output](docs/assets/main_pipeline_view_results_step_3.png)
```

## Why This Matters For the Assignment
- Shows end-to-end reproducible workflow.
- Provides evidence-based model comparison.
- Supports clear explanation during presentation/viva.
