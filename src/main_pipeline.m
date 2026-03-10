%% main_pipeline.m
% MASTER SCRIPT - Complete MNIST Digit Recognition Pipeline
%
% This script runs the entire pattern recognition system from start to finish:
% 1. Data Loading & Splitting
% 2. Preprocessing (Normalization)
% 3. Feature Extraction (Raw, PCA, HOG, Hybrid)
% 4. Model Training (k-NN, SVM, Random Forest)
% 5. Model Evaluation (Metrics, Confusion Matrices)
%
% Team Members: Rohan, [Member 2], [Member 3]
% Course: CT104-3-M Pattern Recognition
% Institution: Asia Pacific University (APU)
% Due Date: March 27, 2026

clear; clc; close all;

%% Display Banner
fprintf('\n');
fprintf('=============================================================\n');
fprintf('   MNIST HANDWRITTEN DIGIT RECOGNITION SYSTEM\n');
fprintf('   Pattern Recognition Assignment - CT104-3-M\n');
fprintf('=============================================================\n\n');

%% User Menu - Choose What to Run
fprintf('Select operation:\n');
fprintf('  1 - Run complete pipeline (all steps)\n');
fprintf('  2 - Run individual step\n');
fprintf('  3 - View existing results\n');
fprintf('  0 - Exit\n\n');

choice = input('Enter choice (0-3): ');

switch choice
    case 1
        run_complete_pipeline();
    case 2
        run_individual_step();
    case 3
        view_results();
    case 0
        fprintf('Exiting...\n');
        return;
    otherwise
        fprintf('Invalid choice. Exiting...\n');
        return;
end

%% ========================================
%% Function: Run Complete Pipeline
%% ========================================
function run_complete_pipeline()
fprintf('\n=== Running Complete Pipeline ===\n\n');

% Confirm before starting (takes ~1-2 hours total)
fprintf('This will run all steps and may take 1-2 hours.\n');
confirm = input('Continue? (y/n): ', 's');

if ~strcmpi(confirm, 'y')
    fprintf('Cancelled.\n');
    return;
end

start_time = tic; % Start timer

% Step 1: Load Data
fprintf('\n--- Step 1/5: Loading Data ---\n');
run('src/load_data.m');
fprintf('✓ Data loading complete.\n');

% Step 2: Preprocess
fprintf('\n--- Step 2/5: Preprocessing ---\n');
run('src/preprocess_data.m');
fprintf('✓ Preprocessing complete.\n');

% Step 3: Feature Extraction
fprintf('\n--- Step 3/5: Feature Extraction ---\n');
run('src/extract_features.m');
fprintf('✓ Feature extraction complete.\n');

% Step 4: Model Training
fprintf('\n--- Step 4/5: Model Training ---\n');
run('src/train_models.m');
fprintf('✓ Model training complete.\n');

% Step 5: Evaluation
fprintf('\n--- Step 5/5: Model Evaluation ---\n');
run('src/evaluate_models.m');
fprintf('✓ Evaluation complete.\n');

total_time = toc(start_time);

% Final summary
fprintf('\n===========================================\n');
fprintf('PIPELINE COMPLETE!\n');
fprintf('===========================================\n\n');
fprintf('Total execution time: %.1f minutes\n', total_time/60);
fprintf('All results saved in results/ directory\n\n');
fprintf('Key outputs:\n');
fprintf('  • results/final_report.txt - Performance summary\n');
fprintf('  • results/evaluation_metrics.mat - All metrics\n');
fprintf('  • results/confusion_matrices/ - Confusion matrices\n\n');
end

%% ========================================
%% Function: Run Individual Step
%% ========================================
function run_individual_step()
fprintf('\n=== Run Individual Step ===\n\n');
fprintf('Select step to run:\n');
fprintf('  1 - Load Data\n');
fprintf('  2 - Preprocess Data\n');
fprintf('  3 - Extract Features\n');
fprintf('  4 - Train Models\n');
fprintf('  5 - Evaluate Models\n');
fprintf('  0 - Back\n\n');

step = input('Enter step (0-5): ');

fprintf('\n');

switch step
    case 1
        fprintf('Running: Load Data\n');
        run('src/load_data.m');
    case 2
        fprintf('Running: Preprocess Data\n');
        run('src/preprocess_data.m');
    case 3
        fprintf('Running: Extract Features\n');
        run('src/extract_features.m');
    case 4
        fprintf('Running: Train Models\n');
        run('src/train_models.m');
    case 5
        fprintf('Running: Evaluate Models\n');
        run('src/evaluate_models.m');
    case 0
        fprintf('Returning...\n');
        return;
    otherwise
        fprintf('Invalid step.\n');
end
end

%% ========================================
%% Function: View Existing Results
%% ========================================
function view_results()
fprintf('\n=== View Results ===\n\n');

% Check if results exist
if ~exist('results/evaluate_model/evaluation_metrics.mat', 'file')
    fprintf('No results found. Run evaluation first.\n');
    return;
end

% Load results
load('results/evaluate_model/evaluation_metrics.mat');
load('results/training_results.mat');

% Display summary
fprintf('=== RESULTS SUMMARY ===\n\n');

% Best model
[max_acc, max_idx] = max(test_accuracies(:));
[best_f, best_m] = ind2sub(size(test_accuracies), max_idx);

fprintf('BEST MODEL:\n');
fprintf('  Algorithm: %s\n', model_names{best_m});
fprintf('  Features: %s\n', feature_names{best_f});
fprintf('  Test Accuracy: %.2f%%\n\n', max_acc * 100);

% Test accuracy table
fprintf('TEST ACCURACIES:\n\n');
fprintf('%-15s | %-10s | %-10s | %-15s\n', 'Feature Set', 'k-NN', 'SVM', 'Random Forest');
fprintf('----------------+------------+------------+-----------------\n');
for f = 1:length(feature_names)
    fprintf('%-15s | %9.2f%% | %9.2f%% | %14.2f%%\n', ...
        feature_names{f}, ...
        test_accuracies(f, 1)*100, ...
        test_accuracies(f, 2)*100, ...
        test_accuracies(f, 3)*100);
end
fprintf('\n');

% Show detailed metrics for best model
best_metrics = all_metrics{best_f, best_m};

fprintf('DETAILED METRICS (Best Model):\n\n');
fprintf('Digit | Precision | Recall | F1-Score\n');
fprintf('------+-----------+--------+---------\n');
for digit = 0:9
    fprintf('  %d   |   %.2f%%   | %.2f%% | %.2f%%\n', ...
        digit, ...
        best_metrics.precision(digit+1) * 100, ...
        best_metrics.recall(digit+1) * 100, ...
        best_metrics.f1_score(digit+1) * 100);
end
fprintf('\n');

fprintf('Full report available in: results/final_report.txt\n\n');
end