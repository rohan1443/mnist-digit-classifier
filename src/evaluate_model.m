% Comprehensive Model Evaluation on Test Set
%
% This script evaluates all trained models on the TEST set (never seen before)
% and calculates detailed performance metrics:
% We are considering the following metrics for evaluation:
% - Accuracy
% - Confusion Matrix
% - Precision, Recall, F1-Score (per digit)
% - Sensitivity & Specificity
%
% Input: data/features.mat (test features)
%        results/models/*.mat (trained models)
%        results/training_results.mat (training summary)
% Output: results/evaluation_metrics.mat (all metrics)
%         results/evaluate_model/confusion_matrices/ (visualizations)
%         results/evaluate_model/final_report.txt (text summary)

clear; clc; close all;

%% Setup
fprintf("=== Model Evaluation Pipeline === \n");

% Create the directories for the results if they don't exist
if ~exist('results/evaluate_model/confusion_matrices', 'dir')
    mkdir('results/evaluate_model/confusion_matrices');
end

% Load data
fprintf('Loading the test features and the training models \n');

load('data/feature-extracted/features.mat', 'features_raw_test', 'features_pca50_test', ...
    'features_pca100_test', 'features_hog_test', 'features_hybrid_test', ...
    'test_labels')

% Load the training results summary
load('results/evaluate_model/training_results.mat');

fprintf("Data loaded successfully! \n\n");
fprintf('Test Samples: %d \n', length(test_labels));
fprintf('Number of classes: %d (digits 0-9) \n\n', length(unique(test_labels)));

% Organising the test features to the same structure as did during training for easy iteration

feature_names = {'RAW', 'PCA50', 'PCA100', 'HOG', 'HYBRID'};

feature_test = struct();
feature_test.RAW = features_raw_test;
feature_test.PCA50 = features_pca50_test;
feature_test.PCA100 = features_pca100_test;
feature_test.HOG = features_hog_test;
feature_test.HYBRID = features_hybrid_test;

%% Initialize the storage for all the metrics
% We will store the metrics for the all the feature combinations and models in a structured way for easy access and comparison later

num_features = length(feature_names);
num_models = 3; % KNN, SVM, Random Forest

% Creating a structure to store the results
test_accuracies = zeros(num_features, num_models);
% Each cell will store metrices for each model and feature combination
all_metrics = cell(num_features, num_models); % This will be a cell array where each cell contains a struct with all the metrics for that model and feature combination


model_names = {'KNN', 'SVM', 'Random Forest'};

% =====================
% Evaluate All Models
% =====================

fprintf("Evaluating all models on the test set \n\n")

for f=1:num_features
    fname = feature_names{f};
    X_test = feature_test.(fname); % Get the test features for the current feature set

    fprintf('Feature Set: %s\n', fname);
    fprintf('----------------------------------------\n');

    for m = 1:num_models
        model_name = model_names{m};

        fprintf('Evaluating Model: %s \n', model_name);

        % Load the trained model from the previous stage
        if m == 1 % KNN
            model_file = sprintf('results/models/knn_%s.mat', lower(fname));
            load(model_file, 'knn_final');
            trained_model = knn_final;
        elseif m == 2 % SVM
            model_file = sprintf('results/models/svm_%s.mat', lower(fname));
            load(model_file, 'svm_model');
            trained_model = svm_model;
        elseif m == 3 % Random Forest
            model_file = sprintf('results/models/rf_%s.mat', lower(fname));
            load(model_file, 'rf_final');
            trained_model = rf_final;
        end

        % Predict the labels on the test set using the loaded model
        if m == 3 % Random Forest uses predict function
            predictions_cell = predict(trained_model, X_test);
            predictions = str2double(predictions_cell);
        else % KNN and SVM use predict function that returns numeric labels
            predictions = predict(trained_model, X_test);
        end

        % Calculate metrics
        metrics = calculate_metrics(test_labels, predictions); % calculate_metrics: our custom function defined below

        % store results
        test_accuracies(f, m) = metrics.accuracy;
        all_metrics{f, m} = metrics;

        % Display the metrics for the current model and feature set
        fprintf('    Test Accuracy: %.2f%%\n', metrics.accuracy * 100);
        fprintf('    Macro F1-Score: %.2f%%\n', metrics.macro_f1 * 100);
        fprintf('\n');

    end
    fprintf('\n\n');
end

fprintf('All models evaluated!\n\n');

%% ========================================
%% Detailed Analysis of Best Model
%% ========================================

fprintf('========================================\n');
fprintf('BEST MODEL DETAILED ANALYSIS\n');
fprintf('========================================\n\n');

% Find best model on test set
[max_test_acc, max_idx] = max(test_accuracies(:));
[best_f, best_m] = ind2sub(size(test_accuracies), max_idx);

fprintf('Best Model Configuration:\n');
fprintf('  Algorithm: %s\n', model_names{best_m});
fprintf('  Features: %s\n', feature_names{best_f});
fprintf('  Test Accuracy: %.2f%%\n\n', max_test_acc * 100);

% Get best model's detailed metrics
best_metrics = all_metrics{best_f, best_m};

fprintf('Detailed Metrics (Best Model):\n');
fprintf('  Overall Accuracy: %.2f%%\n', best_metrics.accuracy * 100);
fprintf('  Macro Precision: %.2f%%\n', best_metrics.macro_precision * 100);
fprintf('  Macro Recall: %.2f%%\n', best_metrics.macro_recall * 100);
fprintf('  Macro F1-Score: %.2f%%\n\n', best_metrics.macro_f1 * 100);


% Per-digit performance
fprintf('Per-Digit Performance:\n');
fprintf('Digit | Precision | Recall | F1-Score | Sensitivity | Specificity\n');
fprintf('------+-----------+--------+----------+-------------+-------------\n');
for digit = 0:9
    fprintf('  %d   |   %.2f%%   | %.2f%% |  %.2f%%   |   %.2f%%     |   %.2f%%\n', ...
        digit, ...
        best_metrics.precision(digit+1) * 100, ...
        best_metrics.recall(digit+1) * 100, ...
        best_metrics.f1_score(digit+1) * 100, ...
        best_metrics.sensitivity(digit+1) * 100, ...
        best_metrics.specificity(digit+1) * 100);
end
fprintf('\n');

%% ========================================
%% Confusion Matrix Visualization
%% ========================================

fprintf('Creating confusion matrix visualizations...\n\n');

% Create confusion matrix for best model
figure('Name', 'Best Model Confusion Matrix', 'Position', [100, 100, 800, 700]);

% Plot confusion matrix
% confusionchart: creates interactive confusion matrix visualization
cm = confusionchart(test_labels, best_metrics.predictions);
cm.Title = sprintf('Confusion Matrix - %s with %s Features', ...
    model_names{best_m}, feature_names{best_f});
cm.RowSummary = 'row-normalized'; % Shows percentage per true class
cm.ColumnSummary = 'column-normalized'; % Shows percentage per predicted class
% Normalized view helps identify which digits are commonly confused

% Save figure
saveas(gcf, sprintf('results/evaluate_model/confusion_matrices/best_model_%s_%s.png', ...
    lower(model_names{best_m}), lower(feature_names{best_f})));
% gcf: get current figure
% saveas: saves figure to file

%% Create confusion matrices for all models (optional - can be slow)
fprintf('Creating confusion matrices for all models...\n');

for f = 1:num_features
    for m = 1:num_models
        metrics = all_metrics{f, m};

        figure('Visible', 'off'); % Don't display (faster), just save
        % 'Visible', 'off': creates figure in background

        cm = confusionchart(test_labels, metrics.predictions);
        cm.Title = sprintf('%s - %s Features', model_names{m}, feature_names{f});
        cm.RowSummary = 'row-normalized';
        cm.ColumnSummary = 'column-normalized';

        filename = sprintf('results/evaluate_model/confusion_matrices/%s_%s.png', ...
            lower(model_names{m}), lower(feature_names{f}));
        saveas(gcf, filename);
        close(gcf); % Close figure to free memory
    end
end

fprintf('All confusion matrices saved!\n\n');

%% ========================================
%% Comparison: Validation vs Test Accuracy
%% ========================================

fprintf('========================================\n');
fprintf('VALIDATION vs TEST ACCURACY\n');
fprintf('========================================\n\n');

% Compare to check for overfitting
% If test accuracy << validation accuracy → overfitting
% If similar → good generalization

fprintf('Comparison (checking for overfitting):\n\n');
fprintf('%-15s | %-12s | Val Acc | Test Acc | Difference\n', 'Features', 'Model');
fprintf('----------------+--------------+---------+----------+------------\n');

for f = 1:num_features
    for m = 1:num_models
        val_acc = results_table(f, m); % From training
        test_acc = test_accuracies(f, m);
        diff = (val_acc - test_acc) * 100;

        fprintf('%-15s | %-12s | %6.2f%% | %7.2f%% | %+7.2f%%\n', ...
            feature_names{f}, model_names{m}, ...
            val_acc * 100, test_acc * 100, diff);
    end
end

fprintf('\n');
fprintf('Note: Small positive difference (0-2%%) is normal.\n');
fprintf('      Large difference (>5%%) may indicate overfitting.\n\n');

%% ========================================
%% Visualize Overall Results
%% ========================================

fprintf('Creating performance comparison charts...\n');

% Chart 1: Test Accuracy Comparison (all models)
figure('Name', 'Test Accuracy Comparison', 'Position', [100, 100, 1000, 600]);

bar(test_accuracies * 100);
set(gca, 'XTickLabel', feature_names);
xlabel('Feature Set');
ylabel('Test Accuracy (%)');
title('Test Set Performance - All Models');
legend(model_names, 'Location', 'southeast');
grid on;
ylim([90 100]); % Zoom to relevant range

% Add accuracy values on bars
for i = 1:size(test_accuracies, 1)
    for j = 1:size(test_accuracies, 2)
        text(i + (j-2)*0.25, test_accuracies(i,j)*100 + 0.3, ...
            sprintf('%.1f', test_accuracies(i,j)*100), ...
            'HorizontalAlignment', 'center', 'FontSize', 8);
    end
end

saveas(gcf, 'results/evaluate_model/test_accuracy_comparison.png');

% Chart 2: Best Model - Per-Digit F1-Scores
figure('Name', 'Per-Digit Performance', 'Position', [100, 100, 800, 500]);

bar(0:9, best_metrics.f1_score * 100);
xlabel('Digit');
ylabel('F1-Score (%)');
title(sprintf('Per-Digit F1-Scores - %s with %s', ...
    model_names{best_m}, feature_names{best_f}));
grid on;
ylim([90 100]);
xticks(0:9);

saveas(gcf, 'results/evaluate_model/per_digit_performance.png');

%% ========================================
%% Generate Text Report for Documentation
%% ========================================

fprintf('Generating final report...\n');

% Create text file with all results (useful for copy-paste into documentation)
report_file = 'results/evaluate_model/final_report.txt';
fid = fopen(report_file, 'w'); % fopen: opens file for writing
% 'w': write mode (creates new file or overwrites existing)

% Write header
fprintf(fid, '==========================================================\n');
fprintf(fid, 'MNIST HANDWRITTEN DIGIT RECOGNITION - EVALUATION REPORT\n');
fprintf(fid, '==========================================================\n\n');
fprintf(fid, 'Date: %s\n\n', datestr(now)); % datestr: formats current date/time

% Test set summary
fprintf(fid, 'TEST SET RESULTS:\n');
fprintf(fid, '-----------------\n');
fprintf(fid, 'Test samples: %d\n', length(test_labels));
fprintf(fid, 'Classes: 10 (digits 0-9)\n\n');

% Overall results table
fprintf(fid, 'OVERALL PERFORMANCE (Test Accuracy):\n\n');
fprintf(fid, '%-15s | %-10s | %-10s | %-15s\n', 'Feature Set', 'k-NN', 'SVM', 'Random Forest');
fprintf(fid, '----------------+------------+------------+-----------------\n');
for f = 1:num_features
    fprintf(fid, '%-15s | %9.2f%% | %9.2f%% | %14.2f%%\n', ...
        feature_names{f}, ...
        test_accuracies(f, 1)*100, ...
        test_accuracies(f, 2)*100, ...
        test_accuracies(f, 3)*100);
end
fprintf(fid, '\n');

% Best model details
fprintf(fid, 'BEST MODEL:\n');
fprintf(fid, '-----------\n');
fprintf(fid, 'Algorithm: %s\n', model_names{best_m});
fprintf(fid, 'Features: %s\n', feature_names{best_f});
fprintf(fid, 'Test Accuracy: %.2f%%\n\n', max_test_acc * 100);

% Detailed metrics
fprintf(fid, 'DETAILED METRICS (Best Model):\n');
fprintf(fid, '------------------------------\n');
fprintf(fid, 'Macro Precision: %.2f%%\n', best_metrics.macro_precision * 100);
fprintf(fid, 'Macro Recall: %.2f%%\n', best_metrics.macro_recall * 100);
fprintf(fid, 'Macro F1-Score: %.2f%%\n\n', best_metrics.macro_f1 * 100);

% Per-digit performance
fprintf(fid, 'PER-DIGIT PERFORMANCE:\n');
fprintf(fid, '---------------------\n');
fprintf(fid, 'Digit | Precision | Recall | F1-Score | Sensitivity | Specificity\n');
fprintf(fid, '------+-----------+--------+----------+-------------+-------------\n');
for digit = 0:9
    fprintf(fid, '  %d   |   %.2f%%   | %.2f%% |  %.2f%%   |   %.2f%%     |   %.2f%%\n', ...
        digit, ...
        best_metrics.precision(digit+1) * 100, ...
        best_metrics.recall(digit+1) * 100, ...
        best_metrics.f1_score(digit+1) * 100, ...
        best_metrics.sensitivity(digit+1) * 100, ...
        best_metrics.specificity(digit+1) * 100);
end

fclose(fid); % Close file
fprintf('Report saved to %s\n\n', report_file);

%% Save all metrics
fprintf('Saving evaluation metrics...\n');

save('results/evaluate_model/evaluation_metrics.mat', ...
    'test_accuracies', 'all_metrics', ...
    'best_f', 'best_m', 'best_metrics', ...
    'feature_names', 'model_names');

fprintf('Metrics saved to results/evaluate_model/evaluation_metrics.mat\n\n');

%% ========================================
%% Final Summary
%% ========================================

fprintf('========================================\n');
fprintf('EVALUATION COMPLETE!\n');
fprintf('========================================\n\n');

fprintf('Summary:\n');
fprintf('• Evaluated 15 model-feature combinations on test set\n');
fprintf('• Best: %s with %s features (%.2f%% accuracy)\n', ...
    model_names{best_m}, feature_names{best_f}, max_test_acc * 100);
fprintf('• Generated confusion matrices for all models\n');
fprintf('• Created detailed performance report\n');
fprintf('• All results saved in results/evaluate_model/ directory\n\n');

fprintf('Files created:\n');
fprintf('  - results/evaluate_model/evaluation_metrics.mat\n');
fprintf('  - results/evaluate_model/final_report.txt\n');
fprintf('  - results/evaluate_model/confusion_matrices/*.png\n');
fprintf('  - results/evaluate_model/test_accuracy_comparison.png\n');
fprintf('  - results/evaluate_model/per_digit_performance.png\n\n');


%% ========================================
%% Helper Function: Calculate All Metrics
%% ========================================

function metrics = calculate_metrics(true_labels, predictions)
% Calculate comprehensive evaluation metrics
% Input: true_labels - actual labels
%        predictions - model predictions
% Output: metrics - structure with all calculated metrics

% Store predictions for confusion matrix later
metrics.predictions = predictions;

% Overall accuracy
% Accuracy = (correct predictions) / (total predictions)
metrics.accuracy = sum(predictions == true_labels) / length(true_labels);

% Get unique classes (digits 0-9)
classes = unique(true_labels);
num_classes = length(classes);

% Initialize arrays for per-class metrics
metrics.precision = zeros(num_classes, 1);
metrics.recall = zeros(num_classes, 1);
metrics.f1_score = zeros(num_classes, 1);
metrics.sensitivity = zeros(num_classes, 1); % Same as recall
metrics.specificity = zeros(num_classes, 1);

% Calculate metrics for each digit
for i = 1:num_classes
    digit = classes(i);

    % True Positives: correctly predicted as this digit
    TP = sum((predictions == digit) & (true_labels == digit));

    % False Positives: incorrectly predicted as this digit
    FP = sum((predictions == digit) & (true_labels ~= digit));

    % False Negatives: this digit predicted as something else
    FN = sum((predictions ~= digit) & (true_labels == digit));

    % True Negatives: correctly predicted as NOT this digit
    TN = sum((predictions ~= digit) & (true_labels ~= digit));

    % Precision: Of all predicted as this digit, how many were correct?
    % Precision = TP / (TP + FP)
    if (TP + FP) > 0
        metrics.precision(i) = TP / (TP + FP);
    else
        metrics.precision(i) = 0;
    end

    % Recall (Sensitivity): Of all actual this digit, how many did we find?
    % Recall = TP / (TP + FN)
    if (TP + FN) > 0
        metrics.recall(i) = TP / (TP + FN);
        metrics.sensitivity(i) = metrics.recall(i); % Same thing
    else
        metrics.recall(i) = 0;
        metrics.sensitivity(i) = 0;
    end

    % F1-Score: Harmonic mean of precision and recall
    % F1 = 2 * (Precision * Recall) / (Precision + Recall)
    if (metrics.precision(i) + metrics.recall(i)) > 0
        metrics.f1_score(i) = 2 * (metrics.precision(i) * metrics.recall(i)) / ...
            (metrics.precision(i) + metrics.recall(i));
    else
        metrics.f1_score(i) = 0;
    end

    % Specificity: Of all NOT this digit, how many did we correctly identify?
    % Specificity = TN / (TN + FP)
    if (TN + FP) > 0
        metrics.specificity(i) = TN / (TN + FP);
    else
        metrics.specificity(i) = 0;
    end
end

% Macro averages (average across all classes, treats each class equally)
metrics.macro_precision = mean(metrics.precision);
metrics.macro_recall = mean(metrics.recall);
metrics.macro_f1 = mean(metrics.f1_score);

end