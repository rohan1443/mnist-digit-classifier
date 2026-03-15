# MNIST Handwritten Digit Recognition System

**Course:** CT104-3-M Pattern Recognition  
**Institution:** Asia Pacific University of Technology & Innovation (APU)  
**Assignment:** Advanced Pattern Recognition for Handwritten Digits

A comprehensive machine learning system for classifying handwritten digits (0-9) using the MNIST dataset with state-of-the-art feature extraction and multiple classification algorithms.

## 🎓 Team Members
- **LOH HOI PING**
- **TEE MUN CHUN**
- **ROHAN MAZUMDAR**

## 📊 Project Overview

This project implements a complete end-to-end pipeline for handwritten digit recognition, achieving **95-98% accuracy** using traditional machine learning approaches. The system includes advanced preprocessing, multiple feature extraction techniques, and three state-of-the-art classification models.

### Key Features
✅ Comprehensive data preprocessing pipeline  
✅ Multiple feature extraction methods (PCA, HOG, LBP, Statistical)  
✅ Three optimized classification models (SVM, Random Forest, k-NN)  
✅ Extensive evaluation metrics and visualizations  
✅ Complete logging and reproducible results  
✅ Fully executable in VS Code with MATLAB extension  

## 📁 Project Structure

```
mnist-digit-classifier/
├── data/
│   ├── csv/                    # Raw MNIST CSV files
│   ├── loaded/                 # Processed .mat files
│   ├── preprocessed/           # Normalized & standardized data
│   └── features/               # Extracted feature sets
├── src/
│   ├── load_data.m             # Data loading & train/val/test split
│   ├── preprocess_data.m       # Advanced preprocessing pipeline
│   ├── extract_features.m      # Multi-method feature extraction
│   ├── train_model.m           # Train SVM, RF, and k-NN models
│   ├── evaluate_model.m        # Comprehensive model evaluation
│   ├── demo_app.m              # Upload-image demo UI (uses saved models)
│   ├── demo_predict_digit.m    # CLI helper for one-image prediction
│   ├── prepare_demo_assets.m   # Generates *.jpg/*.gif/*.png test files
│   └── main_pipeline.m         # Complete workflow orchestrator
├── models/                     # Trained model files (.mat)
├── results/                    # Logs, plots, and analysis
├── README.md                   # This file
└── Pattern Recognition -Assignment - APUMP2601AI.pdf
```

## 🔬 Methodology

### 1. Data Preparation (`load_data.m`)
- **Dataset:** MNIST (70,000 images: 60,000 train + 10,000 test)
- **Format:** 28×28 grayscale images (784 features)
- **Split:** 80% train (48,000), 20% validation (12,000), test (10,000)
- **Output:** Balanced dataset with quality checks

### 2. Preprocessing (`preprocess_data.m`)
Advanced preprocessing techniques based on research best practices:
- **Normalization:** Min-Max scaling [0, 255] → [0, 1]
- **Standardization:** Zero mean, unit variance (z-score)
- **Enhancement (Optional):** CLAHE for contrast improvement
- **Noise Reduction (Optional):** Gaussian smoothing (σ=0.5)
- **Quality Checks:** NaN/Inf detection, balance verification

### 3. Feature Extraction (`extract_features.m`)
Six different feature representations:

| Feature Type | Dimensions | Best For | Description |
|--------------|------------|----------|-------------|
| **Raw Pixels** | 784 | k-NN, NN | Direct normalized pixel values |
| **PCA** | ~95 | SVM, NN | Dimensionality reduction (95% variance) |
| **HOG** | Variable | SVM, RF | Histogram of Oriented Gradients |
| **LBP** | 256 | SVM, k-NN | Local Binary Patterns (texture) |
| **Statistical** | 14 | Ensemble | Moments, symmetry, density, entropy |
| **Combined** | HOG+LBP+Stats | SVM, RF | Concatenated feature fusion |

### 4. Model Training (`train_model.m`)
Three state-of-the-art classification algorithms:

#### **Model 1: Support Vector Machine (SVM)**
- **Algorithm:** ECOC with RBF kernel
- **Features:** HOG (optimal for edge detection)
- **Hyperparameters:** Auto-scaled kernel, BoxConstraint=1
- **Expected Accuracy:** ~98%
- **Reference:** Cortes & Vapnik (1995)

#### **Model 2: Random Forest**
- **Algorithm:** Bootstrap Aggregated Decision Trees
- **Features:** Combined (HOG+LBP+Statistical)
- **Configuration:** 100 trees, min leaf size=5
- **Expected Accuracy:** ~97%
- **Reference:** Breiman (2001)

#### **Model 3: k-Nearest Neighbors (k-NN)**
- **Algorithm:** Distance-based classification
- **Features:** Raw pixels (normalized)
- **Optimization:** Cross-validation for k selection
- **Expected Accuracy:** ~96-97%
- **Reference:** Cover & Hart (1967)

### 5. Evaluation (`evaluate_model.m`)
Comprehensive performance analysis:
- **Metrics:** Accuracy, Precision, Recall, F1-Score
- **Visualization:** Confusion matrices, ROC curves
- **Analysis:** Per-class performance, error patterns
- **Comparison:** Model ranking and recommendations

## 🚀 Getting Started

### Prerequisites
- **MATLAB** (R2020a or later) with toolboxes:
  - Statistics and Machine Learning Toolbox
  - Image Processing Toolbox
  - Computer Vision Toolbox
- **VS Code** (recommended) with MATLAB extension
- **Git** for version control

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rohan1443/mnist-digit-classifier.git
   cd mnist-digit-classifier
   ```

2. **Download MNIST Dataset**
   - Download from: [Kaggle MNIST CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
   - Place files in `data/csv/`:
     - `mnist_train.csv`
     - `mnist_test.csv`

3. **Verify directory structure**
   ```
   mnist-digit-classifier/
   ├── data/csv/
   │   ├── mnist_train.csv  ✓
   │   └── mnist_test.csv   ✓
   ```

## ▶️ How to Run

### Option 1: Complete Pipeline (Recommended)

**In VS Code:**
1. Open `src/main_pipeline.m`
2. Press `F5` or click the ▶️ Run button
3. Wait for completion (~10-30 minutes depending on hardware)

**In MATLAB:**
```matlab
cd('path/to/mnist-digit-classifier')
run('src/main_pipeline.m')
```

### Option 2: Individual Stages

Run stages sequentially:

```matlab
% Stage 1: Load Data
run('src/load_data.m')

% Stage 2: Preprocessing
run('src/preprocess_data.m')

% Stage 3: Feature Extraction
run('src/extract_features.m')

% Stage 4: Model Training
run('src/train_model.m')

% Stage 5: Evaluation
run('src/evaluate_model.m')
```

### Option 3: Run in VS Code (Interactive)

1. Open VS Code in project folder
2. Install **MATLAB extension** by MathWorks
3. Open any `.m` file in `src/`
4. Use Command Palette (`Ctrl+Shift+P`):
   - `MATLAB: Change Current Folder` → Select project root
   - `MATLAB: Run File` or press `F5`

### Option 4: Run Local Upload Demo (No Retraining)

If models already exist in `models/`, run only the demo:

```matlab
cd('path/to/mnist-digit-classifier')
run('src/demo_app.m')
```

The app allows users to upload handwritten digit images and returns:
- Predicted digit by each model (SVM, Random Forest, k-NN)
- Confidence score (%) for each model
- Processed 28x28 digit preview used for inference

Optional CLI mode:

```matlab
result = demo_predict_digit('demo/test_images/digit_7_01.png');
```

Generate submission test images in multiple formats:

```matlab
prepare_demo_assets();
```

## 📈 Expected Results

### Performance Benchmarks

| Model | Validation Acc. | Test Acc. | Training Time | Inference Speed |
|-------|----------------|-----------|---------------|-----------------|
| **SVM** | ~98.0% | ~97.8% | ~60-120s | Fast (0.5s) |
| **Random Forest** | ~97.5% | ~97.3% | ~120-180s | Medium (1.5s) |
| **k-NN** | ~96.8% | ~96.5% | ~10s | Slow (5-10s) |

*Times measured on standard laptop (i5 processor, 8GB RAM)*

### Output Files

After running the complete pipeline:

**Data Files:**
- `data/loaded/mnist_data.mat` - Split datasets
- `data/preprocessed/mnist_preprocessed.mat` - Standardized data
- `data/features/*.mat` - All feature sets

**Models:**
- `models/svm_model.mat` - Trained SVM
- `models/rf_model.mat` - Trained Random Forest
- `models/knn_model.mat` - Trained k-NN

**Results & Visualizations:**
- `results/load_data_log.txt` - Data loading summary
- `results/preprocessing_log.txt` - Preprocessing details
- `results/feature_extraction_log.txt` - Feature statistics
- `results/training_log.txt` - Training metrics
- `results/evaluation_log.txt` - Final performance report
- `results/*.png` - Various visualizations

## 📊 Visualizations Generated

1. **Sample digits** - Random dataset examples
2. **Preprocessing comparison** - Original vs. processed images
3. **PCA analysis** - Variance explained, component visualization
4. **Feature distributions** - Histogram comparisons
5. **Training summary** - Accuracy, time, speed comparison
6. **Confusion matrices** - Per-model classification results
7. **Model comparison** - Comprehensive performance metrics
8. **Error analysis** - Misclassified examples
9. **Per-class performance** - Precision/Recall/F1 trends

## 🔧 Configuration Options

Edit `main_pipeline.m` to customize execution:

```matlab
% Run specific stages
run_load_data = true;
run_preprocessing = true;
run_feature_extraction = true;
run_training = true;
run_evaluation = true;

% Quick mode (smaller dataset for testing)
quick_mode = false;  % Set to true for faster testing
```

## 📚 Technical References

1. **LeCun, Y., et al. (1998).** "Gradient-based learning applied to document recognition." *Proceedings of the IEEE*.

2. **Cortes, C., & Vapnik, V. (1995).** "Support-vector networks." *Machine Learning*, 20(3), 273-297.

3. **Breiman, L. (2001).** "Random forests." *Machine Learning*, 45(1), 5-32.

4. **Cover, T., & Hart, P. (1967).** "Nearest neighbor pattern classification." *IEEE Transactions on Information Theory*.

5. **Dalal, N., & Triggs, B. (2005).** "Histograms of oriented gradients for human detection." *CVPR*.

6. **Ojala, T., et al. (2002).** "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns." *IEEE TPAMI*.

## 🐛 Troubleshooting

### Common Issues

**Issue:** `File not found` error
- **Solution:** Ensure you're running from project root directory
- Use: `cd('path/to/mnist-digit-classifier')`

**Issue:** `Undefined function or variable`
- **Solution:** Install required MATLAB toolboxes
- Check: `ver` in MATLAB command window

**Issue:** Out of memory error
- **Solution:** Close other applications, reduce dataset size in quick_mode

**Issue:** Figures not displaying in VS Code
- **Solution:** Figures will auto-save to `results/` folder

**Issue:** Slow performance
- **Solution:** Enable quick_mode for testing, use smaller feature sets

## 📝 Assignment Compliance

This implementation fulfills all requirements:

✅ **1.3 Dataset Selection & Preparation**
- MNIST dataset loaded and validated
- Train/validation/test split implemented
- Data quality checks performed

✅ **1.4 Pre-processing**
- Normalization & standardization
- Contrast enhancement (CLAHE)
- Noise reduction techniques
- Comprehensive preprocessing pipeline

✅ **1.5 Feature Extraction**
- PCA dimensionality reduction
- HOG edge features
- LBP texture analysis
- Statistical feature engineering
- Combined feature fusion

✅ **1.6 Model Training & Evaluation**
- Three state-of-the-art models implemented
- Proper train/validation/test methodology
- Comprehensive evaluation metrics
- Detailed performance analysis

✅ **Documentation & Logging**
- All stages generate `.txt` logs
- Visualizations saved as `.png` files
- Complete execution tracking

## 🎯 Future Enhancements

- [ ] Deep learning models (CNN)
- [ ] Data augmentation pipeline
- [ ] Real-time digit recognition
- [ ] Model ensemble methods
- [ ] Web/mobile deployment
- [ ] Transfer learning experiments

## 📧 Contact

For questions or issues:
- **Repository:** [github.com/rohan1443/mnist-digit-classifier](https://github.com/rohan1443/mnist-digit-classifier)
- **Course:** CT104-3-M Pattern Recognition
- **Institution:** APU Malaysia

## 📄 License

This project is created for educational purposes as part of the Pattern Recognition course at APU.

---

**Last Updated:** March 2026  
**Version:** 1.0  
**Status:** ✅ Complete and Production-Ready
