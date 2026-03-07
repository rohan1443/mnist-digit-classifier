%% main_pipeline.m
% Runs the full pattern recognition pipeline (1.3--1.8).
% Run from project root (mnist-digit-classifier). All steps write reports
% to results/reports/; the final step generates results_analysis_report.txt.
%
% Steps:
%   1. Preprocess data (load, normalize, split train/val/test)
%   2. Feature extraction (raw + PCA)
%   3. Train models (Naive Bayes, k-NN, LDA, Random Forest, SVM)
%   4. Evaluate on test set (accuracy, precision, recall, confusion matrix)
%   5. Generate results analysis report

function main_pipeline()
rootDir = get_project_root();
cd(rootDir);
addpath(fullfile(rootDir, 'src'));

fprintf('========== MNIST Pattern Recognition Pipeline ==========\n');
fprintf('Project root: %s\n\n', rootDir);

%% 1. Preprocessing
fprintf('Step 1: Preprocessing...\n');
[~, ~, ~, ~, ~, ~, prepReport] = preprocess_data();
fprintf('  Report: %s\n\n', prepReport);

%% 2. Feature extraction
fprintf('Step 2: Feature extraction...\n');
[~, ~, ~, ~, ~, ~, featReport] = extract_features();
fprintf('  Report: %s\n\n', featReport);

%% 3. Training (may take several minutes; Random Forest is slowest)
fprintf('Step 3: Training models (Naive Bayes, k-NN, LDA, Random Forest, SVM)...\n');
[~, trainReport] = train_model();
fprintf('  Report: %s\n\n', trainReport);

%% 4. Evaluation
fprintf('Step 4: Evaluating on test set...\n');
[~, evalReport] = evaluate_model();
fprintf('  Report: %s\n\n', evalReport);

%% 5. Results analysis (compiled report from all steps)
fprintf('Step 5: Generating results analysis...\n');
analysisReport = generate_results_analysis();
fprintf('  Report: %s\n\n', analysisReport);

fprintf('========== Pipeline complete ==========\n');
fprintf('All reports saved under results/reports/\n');
fprintf('Summary: results/reports/results_analysis_report.txt\n');
end

function rootDir = get_project_root()
[cwd, name, ~] = fileparts(pwd);
if strcmp(name, 'src'), rootDir = cwd; else, rootDir = pwd; end
end

%% Run pipeline when this file is executed (e.g. F5 in Cursor/VSCode with MATLAB)
main_pipeline();
