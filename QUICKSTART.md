# MNIST Digit Recognition - Quick Start Guide

## 📋 Prerequisites Checklist

Before running the pipeline, ensure you have:

- [ ] MATLAB R2020a or later installed
- [ ] Required MATLAB Toolboxes:
  - [ ] Statistics and Machine Learning Toolbox
  - [ ] Image Processing Toolbox  
  - [ ] Computer Vision Toolbox
- [ ] VS Code with MATLAB extension (optional but recommended)
- [ ] MNIST CSV files downloaded and placed in `data/csv/`

### Check Your MATLAB Toolboxes

Run this in MATLAB command window:
```matlab
ver
```

Look for these in the output:
- Statistics and Machine Learning Toolbox
- Image Processing Toolbox
- Computer Vision Toolbox

## 📥 Step 1: Download MNIST Dataset

1. Go to: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
2. Download the dataset (requires Kaggle account)
3. Extract and place these files in `data/csv/`:
   - `mnist_train.csv` (60,000 samples)
   - `mnist_test.csv` (10,000 samples)

### Verify File Placement

Your directory should look like:
```
mnist-digit-classifier/
├── data/
│   └── csv/
│       ├── mnist_train.csv  ← Should be here
│       └── mnist_test.csv   ← Should be here
└── src/
    └── ... (code files)
```

## 🚀 Step 2: Run the Complete Pipeline

### Method A: Using VS Code (Recommended)

1. **Open VS Code**
2. **Open the project folder**: `File > Open Folder` → Select `mnist-digit-classifier`
3. **Open the terminal**: `` Ctrl+` `` (backtick)
4. **Navigate to project root** (if not already there)
5. **Open** `src/main_pipeline.m`
6. **Run**: Press `F5` or click the ▶️ button in top-right corner

The pipeline will execute all 5 stages automatically!

### Method B: Using MATLAB Desktop

1. **Launch MATLAB**
2. **Navigate to project folder**:
   ```matlab
   cd('C:\Users\Hannah\Documents\GitHub\mnist-digit-classifier')
   ```
3. **Run the pipeline**:
   ```matlab
   run('src/main_pipeline.m')
   ```

## ⏱️ Expected Execution Time

| Stage | Duration | Description |
|-------|----------|-------------|
| 1. Load Data | 1-2 min | Read CSV, split datasets |
| 2. Preprocessing | 2-3 min | Normalize, standardize |
| 3. Feature Extraction | 5-10 min | HOG, LBP, PCA, statistical |
| 4. Model Training | 10-20 min | Train SVM, RF, k-NN |
| 5. Evaluation | 2-3 min | Test and analyze |
| **Total** | **20-40 min** | Full pipeline |

*Times vary based on hardware*

## 📊 What to Expect

### Console Output

You'll see progress messages like:
```
========================================
  MNIST Digit Recognition Pipeline
  Pattern Recognition Assignment
========================================

✓ Directory structure verified

========================================
STAGE 1: DATA LOADING
========================================
Executing load_data.m...

Loading MNIST dataset...
Files loaded successfully!
...
```

### Generated Files

After completion, check these directories:

**`data/loaded/`** - Processed datasets
- `mnist_data.mat`

**`data/preprocessed/`** - Cleaned data
- `mnist_preprocessed.mat`
- `mnist_normalized.mat`

**`data/features/`** - Feature sets
- `features_raw.mat`
- `features_pca.mat`
- `features_hog.mat`
- `features_lbp.mat`
- `features_statistical.mat`
- `features_combined.mat`

**`models/`** - Trained models
- `svm_model.mat`
- `rf_model.mat`
- `knn_model.mat`

**`results/`** - Logs & Visualizations
- `load_data_log.txt`
- `preprocessing_log.txt`
- `feature_extraction_log.txt`
- `training_log.txt`
- `evaluation_log.txt`
- `pipeline_summary.txt`
- Multiple `.png` visualization files

## 🔍 Reviewing Results

### 1. Check Overall Performance

Open: `results/evaluation_log.txt`

Look for:
```
=== MODEL 1: SVM ===
Overall Performance:
  Accuracy: 97.85%
  Precision (avg): 97.82%
  Recall (avg): 97.79%
  F1-Score (avg): 97.80%
```

### 2. View Visualizations

Open these PNG files in `results/`:
- `confusion_matrices.png` - Classification accuracy per digit
- `model_comparison.png` - Performance metrics comparison
- `training_summary.png` - Training time and accuracy
- `per_class_performance.png` - Detailed per-digit analysis

### 3. Check Training Summary

Open: `results/training_log.txt`

Contains:
- Model configurations
- Validation accuracies
- Training times
- Recommendations

## 🎯 Running Individual Stages

If you want to run stages separately:

```matlab
% Make sure you're in project root
cd('C:\Users\Hannah\Documents\GitHub\mnist-digit-classifier')

% Stage 1: Load and split data
run('src/load_data.m')

% Stage 2: Preprocess data
run('src/preprocess_data.m')

% Stage 3: Extract features
run('src/extract_features.m')

% Stage 4: Train models
run('src/train_model.m')

% Stage 5: Evaluate models
run('src/evaluate_model.m')
```

**Important:** Run stages in order! Each stage depends on the previous one.

## 🖼️ Local Demo Application (No Retraining Needed)

Use this when you already have trained models in `models/` and want a live upload demo.

### 1) Launch the interactive demo UI

```matlab
cd('C:\Users\Hannah\Documents\GitHub\mnist-digit-classifier')
run('src/demo_app.m')
```

### 2) Demo flow

1. Click **Upload Digit Image**.
2. Select any handwritten digit image (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tif`).
3. The app will show:
   - Uploaded image
   - Processed MNIST-style 28x28 image
   - Prediction and confidence for each model (SVM, Random Forest, k-NN)

### 3) Command-line prediction (optional)

```matlab
result = demo_predict_digit('demo/test_images/digit_7_01.png')
```

### 4) Generate submission test images (`*.jpg`, `*.gif`, `*.png`)

```matlab
prepare_demo_assets();    % default: 3 samples per digit
```

Generated files:
- `demo/test_images/*.png`
- `demo/test_images/*.jpg`
- `demo/test_images/*.gif`
- `demo/test_images/manifest.csv`

This satisfies the requirement to provide test files in image formats.

## 🐛 Common Issues & Solutions

### Issue 1: "File not found" Error
```
Error: Unable to read file 'data/csv/mnist_train.csv'
```
**Solution:** Download MNIST CSV files and place in `data/csv/` folder

### Issue 2: "Undefined function or variable"
```
Unrecognized function or variable 'fitcecoc'
```
**Solution:** Install Statistics and Machine Learning Toolbox
- MATLAB → Add-Ons → Get Add-Ons → Search for toolbox

### Issue 3: "Out of Memory"
```
Error: Out of memory
```
**Solution:** 
- Close other applications
- Restart MATLAB
- Enable `quick_mode = true` in `main_pipeline.m` (line 47)

### Issue 4: "Cannot change to directory"
```
Error: Cannot CD to 'path/to/folder'
```
**Solution:** Use absolute path with correct slashes:
```matlab
cd('C:\Users\Hannah\Documents\GitHub\mnist-digit-classifier')
```

### Issue 5: Figures Not Showing in VS Code
**Solution:** This is normal! Figures are automatically saved to `results/` folder as PNG files.

### Issue 6: Very Slow Execution
**Solution:** 
1. Enable quick mode in `main_pipeline.m`:
   ```matlab
   quick_mode = true;  % Line 47
   ```
2. Or run on a computer with better specifications

## 📝 Tips for Best Results

### 1. Ensure Clean Environment
```matlab
clear all; close all; clc;
```

### 2. Check Working Directory
```matlab
pwd  % Should be in mnist-digit-classifier folder
```

### 3. Monitor Progress
- Watch console output for error messages
- Check `results/` folder for log files
- Review generated plots as they appear

### 4. Save Workspace (Optional)
After running pipeline:
```matlab
save('workspace_backup.mat')
```

## 📈 Understanding Your Results

### Good Results Indicators:
✅ Accuracy > 95% on test set  
✅ Balanced precision/recall across all digits  
✅ Low confusion between similar digits (e.g., 3 vs 8)  
✅ Training completes without errors  

### What If Results Are Lower?
- Check data quality in preprocessing logs
- Review feature extraction completeness
- Verify model training convergence
- Check for data leakage or overfitting

## 🎓 Assignment Submission Checklist

Before submitting, ensure you have:

- [ ] All 5 stages executed successfully
- [ ] Log files generated in `results/`
- [ ] Visualizations saved as PNG files
- [ ] Models trained and saved in `models/`
- [ ] Accuracy > 95% achieved
- [ ] Documentation complete (README.md)
- [ ] Code properly commented
- [ ] No hardcoded paths (use relative paths)

## 📧 Need Help?

1. **Check logs first**: Look in `results/*.txt` files
2. **Review error messages**: Copy exact error text
3. **Verify toolboxes**: Run `ver` in MATLAB
4. **Check file paths**: Ensure correct directory structure
5. **Consult README.md**: Full documentation available

## 🎉 Success!

If you see this at the end:
```
✅ All stages completed successfully!

Thank you for using the MNIST Recognition Pipeline!
========================================
```

**Congratulations!** 🎊 Your pipeline ran successfully!

Now:
1. Review `results/evaluation_log.txt` for final accuracy
2. Check `results/confusion_matrices.png` for visual analysis
3. Read `results/pipeline_summary.txt` for complete report
4. Use models in `models/` folder for predictions

---

**Last Updated:** March 2026  
**For:** CT104-3-M Pattern Recognition Assignment  
**Institution:** Asia Pacific University (APU)
