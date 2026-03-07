%% main_pipeline.m
% Complete end-to-end pipeline for MNIST handwritten digit recognition
% Executes all stages: Load → Preprocess → Extract → Train → Evaluate
%
% Course: CT104-3-M Pattern Recognition
% Institution: Asia Pacific University (APU)
% Team: LOH HOI PING, TEE MUN CHUN, ROHAN MAZUMDAR

clc; close all;
fprintf('========================================\n');
fprintf('  MNIST Digit Recognition Pipeline\n');
fprintf('  Pattern Recognition Assignment\n');
fprintf('========================================\n\n');

%% Pipeline Configuration

% Set random seed for reproducibility
rng(42);

% Create all necessary directories
rootDir = pwd;
directories = {'data', 'data/csv', 'data/loaded', 'data/preprocessed', ...
               'data/features', 'models', 'results'};

fprintf('Checking directory structure...\n');
for i = 1:length(directories)
    dir_path = fullfile(rootDir, directories{i});
    if ~exist(dir_path, 'dir')
        mkdir(dir_path);
        fprintf('  Created: %s\n', directories{i});
    end
end
fprintf('✓ Directory structure verified\n\n');

%% Pipeline Execution Options

% Option to run specific stages (set to true/false)
run_load_data = true;
run_preprocessing = true;
run_feature_extraction = true;
run_training = true;
run_evaluation = true;

% Quick mode: Use smaller datasets for faster testing (set to false for full run)
quick_mode = false;

if quick_mode
    fprintf('⚡ QUICK MODE ENABLED - Using reduced dataset for testing\n\n');
end

%% Stage 1: Load Data

if run_load_data
    fprintf('========================================\n');
    fprintf('STAGE 1: DATA LOADING\n');
    fprintf('========================================\n');
    
    stage_start = tic;
    
    % Check if CSV files exist
    train_csv = fullfile(rootDir, 'data', 'csv', 'mnist_train.csv');
    test_csv = fullfile(rootDir, 'data', 'csv', 'mnist_test.csv');
    
    if ~exist(train_csv, 'file') || ~exist(test_csv, 'file')
        fprintf('⚠ WARNING: MNIST CSV files not found!\n');
        fprintf('Please download the dataset from:\n');
        fprintf('https://www.kaggle.com/datasets/oddrationale/mnist-in-csv\n');
        fprintf('And place files in: data/csv/\n');
        error('Required data files not found');
    end
    
    % Execute load_data.m
    fprintf('Executing load_data.m...\n\n');
    run(fullfile(rootDir, 'src', 'load_data.m'));
    
    stage_time = toc(stage_start);
    fprintf('\n✓ Stage 1 completed in %.2f seconds\n', stage_time);
    fprintf('========================================\n\n');
    
    pause(2); % Brief pause between stages
else
    fprintf('⊗ Skipping Stage 1: Data Loading\n\n');
end

%% Stage 2: Preprocessing

if run_preprocessing
    fprintf('========================================\n');
    fprintf('STAGE 2: DATA PREPROCESSING\n');
    fprintf('========================================\n');
    
    stage_start = tic;
    
    % Check if loaded data exists
    if ~exist(fullfile(rootDir, 'data', 'loaded', 'mnist_data.mat'), 'file')
        error('Loaded data not found. Please run Stage 1 first.');
    end
    
    % Execute preprocess_data.m
    fprintf('Executing preprocess_data.m...\n\n');
    run(fullfile(rootDir, 'src', 'preprocess_data.m'));
    
    stage_time = toc(stage_start);
    fprintf('\n✓ Stage 2 completed in %.2f seconds\n', stage_time);
    fprintf('========================================\n\n');
    
    pause(2);
else
    fprintf('⊗ Skipping Stage 2: Preprocessing\n\n');
end

%% Stage 3: Feature Extraction

if run_feature_extraction
    fprintf('========================================\n');
    fprintf('STAGE 3: FEATURE EXTRACTION\n');
    fprintf('========================================\n');
    
    stage_start = tic;
    
    % Check if preprocessed data exists
    if ~exist(fullfile(rootDir, 'data', 'preprocessed', 'mnist_preprocessed.mat'), 'file')
        error('Preprocessed data not found. Please run Stage 2 first.');
    end
    
    % Execute extract_features.m
    fprintf('Executing extract_features.m...\n\n');
    run(fullfile(rootDir, 'src', 'extract_features.m'));
    
    stage_time = toc(stage_start);
    fprintf('\n✓ Stage 3 completed in %.2f seconds\n', stage_time);
    fprintf('========================================\n\n');
    
    pause(2);
else
    fprintf('⊗ Skipping Stage 3: Feature Extraction\n\n');
end

%% Stage 4: Model Training

if run_training
    fprintf('========================================\n');
    fprintf('STAGE 4: MODEL TRAINING\n');
    fprintf('========================================\n');
    
    stage_start = tic;
    
    % Check if features exist
    features_dir = fullfile(rootDir, 'data', 'features');
    if ~exist(features_dir, 'dir') || isempty(dir(fullfile(features_dir, '*.mat')))
        error('Feature data not found. Please run Stage 3 first.');
    end
    
    % Execute train_model.m
    fprintf('Executing train_model.m...\n\n');
    run(fullfile(rootDir, 'src', 'train_model.m'));
    
    stage_time = toc(stage_start);
    fprintf('\n✓ Stage 4 completed in %.2f seconds\n', stage_time);
    fprintf('========================================\n\n');
    
    pause(2);
else
    fprintf('⊗ Skipping Stage 4: Model Training\n\n');
end

%% Stage 5: Model Evaluation

if run_evaluation
    fprintf('========================================\n');
    fprintf('STAGE 5: MODEL EVALUATION\n');
    fprintf('========================================\n');
    
    stage_start = tic;
    
    % Check if trained models exist
    models_dir = fullfile(rootDir, 'models');
    if ~exist(models_dir, 'dir') || isempty(dir(fullfile(models_dir, '*.mat')))
        error('Trained models not found. Please run Stage 4 first.');
    end
    
    % Execute evaluate_model.m
    fprintf('Executing evaluate_model.m...\n\n');
    run(fullfile(rootDir, 'src', 'evaluate_model.m'));
    
    stage_time = toc(stage_start);
    fprintf('\n✓ Stage 5 completed in %.2f seconds\n', stage_time);
    fprintf('========================================\n\n');
else
    fprintf('⊗ Skipping Stage 5: Model Evaluation\n\n');
end

%% Final Summary

fprintf('\n\n');
fprintf('========================================\n');
fprintf('  PIPELINE EXECUTION COMPLETE!\n');
fprintf('========================================\n\n');

% Generate final summary report
fprintf('📊 Summary Report:\n');
fprintf('  Location: results/\n\n');

% List all generated files
results_dir = fullfile(rootDir, 'results');
if exist(results_dir, 'dir')
    results_files = dir(fullfile(results_dir, '*.*'));
    fprintf('  Generated Files:\n');
    for i = 1:length(results_files)
        if ~results_files(i).isdir
            fprintf('    - %s\n', results_files(i).name);
        end
    end
end

fprintf('\n📁 Output Directories:\n');
fprintf('  - data/loaded/        : Loaded and split datasets\n');
fprintf('  - data/preprocessed/  : Normalized and standardized data\n');
fprintf('  - data/features/      : Extracted feature sets\n');
fprintf('  - models/             : Trained model files\n');
fprintf('  - results/            : Logs, plots, and analysis\n');

fprintf('\n📈 Key Results:\n');

% Try to load and display final results
try
    % Load evaluation results if available
    eval_log = fullfile(rootDir, 'results', 'evaluation_log.txt');
    if exist(eval_log, 'file')
        fprintf('  Check evaluation_log.txt for detailed metrics\n');
    end
    
    % Check if models exist and load validation accuracies
    if exist(fullfile(rootDir, 'models', 'svm_model.mat'), 'file')
        load(fullfile(rootDir, 'models', 'svm_model.mat'), 'svm_val_accuracy');
        fprintf('  - SVM Validation Accuracy: %.2f%%\n', svm_val_accuracy);
    end
    
    if exist(fullfile(rootDir, 'models', 'rf_model.mat'), 'file')
        load(fullfile(rootDir, 'models', 'rf_model.mat'), 'rf_val_accuracy');
        fprintf('  - Random Forest Validation Accuracy: %.2f%%\n', rf_val_accuracy);
    end
    
    if exist(fullfile(rootDir, 'models', 'knn_model.mat'), 'file')
        load(fullfile(rootDir, 'models', 'knn_model.mat'), 'knn_val_accuracy');
        fprintf('  - k-NN Validation Accuracy: %.2f%%\n', knn_val_accuracy);
    end
catch
    fprintf('  (Run full pipeline to see results)\n');
end

fprintf('\n✅ All stages completed successfully!\n');
fprintf('\n========================================\n');
fprintf('  Next Steps:\n');
fprintf('========================================\n');
fprintf('1. Review results/ directory for visualizations\n');
fprintf('2. Check log files for detailed metrics\n');
fprintf('3. Analyze confusion matrices\n');
fprintf('4. Consider model deployment options\n\n');

fprintf('Thank you for using the MNIST Recognition Pipeline!\n');
fprintf('========================================\n');

%% Generate Final Pipeline Summary Document

summary_file = fullfile(rootDir, 'results', 'pipeline_summary.txt');
fid = fopen(summary_file, 'w');

fprintf(fid, '================================================================\n');
fprintf(fid, '  MNIST HANDWRITTEN DIGIT RECOGNITION - FINAL REPORT\n');
fprintf(fid, '================================================================\n\n');
fprintf(fid, 'Course: CT104-3-M Pattern Recognition\n');
fprintf(fid, 'Institution: Asia Pacific University (APU)\n');
fprintf(fid, 'Date: %s\n\n', datestr(now));

fprintf(fid, 'TEAM MEMBERS:\n');
fprintf(fid, '  - LOH HOI PING\n');
fprintf(fid, '  - TEE MUN CHUN\n');
fprintf(fid, '  - ROHAN MAZUMDAR\n\n');

fprintf(fid, '================================================================\n');
fprintf(fid, 'PROJECT OVERVIEW\n');
fprintf(fid, '================================================================\n\n');
fprintf(fid, 'Objective:\n');
fprintf(fid, '  Develop a machine learning system for handwritten digit\n');
fprintf(fid, '  recognition using the MNIST dataset with multiple classification\n');
fprintf(fid, '  algorithms and comprehensive feature engineering.\n\n');

fprintf(fid, 'Dataset:\n');
fprintf(fid, '  - Name: MNIST (Modified National Institute of Standards)\n');
fprintf(fid, '  - Total Images: 70,000 (60,000 train + 10,000 test)\n');
fprintf(fid, '  - Image Size: 28x28 pixels (784 features)\n');
fprintf(fid, '  - Classes: 10 (digits 0-9)\n');
fprintf(fid, '  - Data Split: 80%% train, 20%% validation, separate test set\n\n');

fprintf(fid, '================================================================\n');
fprintf(fid, 'METHODOLOGY\n');
fprintf(fid, '================================================================\n\n');

fprintf(fid, '1. Data Preparation (load_data.m):\n');
fprintf(fid, '   - Load MNIST dataset from CSV format\n');
fprintf(fid, '   - Split into training/validation/test sets\n');
fprintf(fid, '   - Verify data integrity and balance\n');
fprintf(fid, '   - Generate sample visualizations\n\n');

fprintf(fid, '2. Preprocessing (preprocess_data.m):\n');
fprintf(fid, '   - Min-Max normalization: [0, 255] → [0, 1]\n');
fprintf(fid, '   - Standardization: Zero mean, unit variance\n');
fprintf(fid, '   - Optional: CLAHE contrast enhancement\n');
fprintf(fid, '   - Optional: Gaussian noise reduction\n');
fprintf(fid, '   - Quality validation checks\n\n');

fprintf(fid, '3. Feature Extraction (extract_features.m):\n');
fprintf(fid, '   a) Raw Pixels (784 features)\n');
fprintf(fid, '   b) PCA - Principal Component Analysis (95%% variance)\n');
fprintf(fid, '   c) HOG - Histogram of Oriented Gradients\n');
fprintf(fid, '   d) LBP - Local Binary Patterns (texture)\n');
fprintf(fid, '   e) Statistical Features (14 custom features)\n');
fprintf(fid, '   f) Combined Features (HOG+LBP+Statistical)\n\n');

fprintf(fid, '4. Model Training (train_model.m):\n');
fprintf(fid, '   Three state-of-the-art algorithms:\n\n');
fprintf(fid, '   a) Support Vector Machine (SVM):\n');
fprintf(fid, '      - Kernel: Radial Basis Function (RBF)\n');
fprintf(fid, '      - Multi-class: Error-Correcting Output Codes\n');
fprintf(fid, '      - Features: HOG\n');
fprintf(fid, '      - Reference: Cortes & Vapnik (1995)\n\n');
fprintf(fid, '   b) Random Forest:\n');
fprintf(fid, '      - Trees: 100 with bootstrap aggregation\n');
fprintf(fid, '      - Features: Combined (HOG+LBP+Statistical)\n');
fprintf(fid, '      - Reference: Breiman (2001)\n\n');
fprintf(fid, '   c) k-Nearest Neighbors (k-NN):\n');
fprintf(fid, '      - Optimal k: Determined via cross-validation\n');
fprintf(fid, '      - Distance: Euclidean\n');
fprintf(fid, '      - Features: Raw pixels (normalized)\n');
fprintf(fid, '      - Reference: Cover & Hart (1967)\n\n');

fprintf(fid, '5. Evaluation (evaluate_model.m):\n');
fprintf(fid, '   - Test set performance measurement\n');
fprintf(fid, '   - Confusion matrices for all models\n');
fprintf(fid, '   - Per-class metrics: Precision, Recall, F1-Score\n');
fprintf(fid, '   - Error analysis and visualization\n');
fprintf(fid, '   - Model comparison and ranking\n\n');

fprintf(fid, '================================================================\n');
fprintf(fid, 'IMPLEMENTATION DETAILS\n');
fprintf(fid, '================================================================\n\n');
fprintf(fid, 'Software Environment:\n');
fprintf(fid, '  - Platform: MATLAB (with Statistics & Machine Learning Toolbox)\n');
fprintf(fid, '  - IDE: Visual Studio Code with MATLAB extension\n');
fprintf(fid, '  - Version Control: Git/GitHub\n\n');

fprintf(fid, 'Best Practices Implemented:\n');
fprintf(fid, '  ✓ Reproducible results (random seed: 42)\n');
fprintf(fid, '  ✓ Proper train/validation/test split\n');
fprintf(fid, '  ✓ Cross-validation for hyperparameter tuning\n');
fprintf(fid, '  ✓ Feature standardization to prevent leakage\n');
fprintf(fid, '  ✓ Comprehensive logging and visualization\n');
fprintf(fid, '  ✓ Modular code structure for maintainability\n\n');

fprintf(fid, '================================================================\n');
fprintf(fid, 'REFERENCES\n');
fprintf(fid, '================================================================\n\n');
fprintf(fid, '[1] LeCun, Y., et al. (1998). Gradient-based learning applied\n');
fprintf(fid, '    to document recognition. Proceedings of the IEEE.\n\n');
fprintf(fid, '[2] Cortes, C., & Vapnik, V. (1995). Support-vector networks.\n');
fprintf(fid, '    Machine Learning, 20(3), 273-297.\n\n');
fprintf(fid, '[3] Breiman, L. (2001). Random forests. Machine Learning,\n');
fprintf(fid, '    45(1), 5-32.\n\n');
fprintf(fid, '[4] Cover, T., & Hart, P. (1967). Nearest neighbor pattern\n');
fprintf(fid, '    classification. IEEE Transactions on Information Theory.\n\n');
fprintf(fid, '[5] Dalal, N., & Triggs, B. (2005). Histograms of oriented\n');
fprintf(fid, '    gradients for human detection. CVPR.\n\n');
fprintf(fid, '[6] Ojala, T., et al. (2002). Multiresolution gray-scale and\n');
fprintf(fid, '    rotation invariant texture classification with local binary\n');
fprintf(fid, '    patterns. IEEE TPAMI.\n\n');

fprintf(fid, '================================================================\n');
fprintf(fid, 'END OF REPORT\n');
fprintf(fid, '================================================================\n');

fclose(fid);

fprintf('\n📄 Final summary document generated: results/pipeline_summary.txt\n\n');
