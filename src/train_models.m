% What We'll Do:
% Train 3 different classifiers on each feature set and compare:
% Models to implement:

% k-NN (k-Nearest Neighbors) - Simple, distance-based
% SVM (Support Vector Machine) - Powerful, finds decision boundaries
% Random Forest - Ensemble method, handles high dimensions well

% Feature sets to test (5 total):

% Raw pixels (784 features)
% PCA-50
% PCA-100
% HOG
% Hybrid (HOG+PCA)

% Total experiments: 3 models × 5 feature sets = 15 combinations

% For Each Model:

% Train on training set
% Tune hyperparameters using validation set
% Save trained model
% Quick validation accuracy check

% Hyperparameters we'll tune:

% k-NN: k value (3, 5, 7, 10)
% SVM: kernel type (linear, RBF), C parameter
% Random Forest: number of trees (50, 100)


% Output:
% results/
% ├── models/
% │   ├── knn_raw.mat
% │   ├── knn_pca50.mat
% │   ├── svm_hybrid.mat
% │   └── ... (15 models total)
% └── training_results.mat (validation accuracies)

% Why This Approach:
% ✅ Shows comparison (required by assignment)
% ✅ Tests multiple algorithms (CLO2 requirement)
% ✅ Hyperparameter tuning (optimization section)
% ✅ Strong evidence for report
% ✅ Can discuss trade-offs in demo

% Estimated Time:

% k-NN: Fast (~2-5 min per feature set)
% SVM: Slower (~5-15 min per feature set)
% Random Forest: Medium (~3-10 min per feature set)

% Total: ~30-60 min to train all 15 models


%% train_models.m
% Train Multiple Classifiers on Different Feature Sets
%
% This script trains 3 different machine learning models:
% 1. k-NN (k-Nearest Neighbors)
% 2. SVM (Support Vector Machine)
% 3. Random Forest
%
% Each model is trained on 5 different feature sets:
% - Raw pixels, PCA-50, PCA-100, HOG, Hybrid (HOG+PCA)
%
% Total: 3 models × 5 feature sets = 15 trained classifiers
%
% Input: data/features.mat (extracted features)
% Output: results/models/*.mat (trained models)
%         results/training_results.mat (validation accuracies)

clear; clc; close all;

%% Setup
fprintf('=== Model Training Pipeline ===\n\n');

% Create results directory if it doesn't exist
if ~exist('results', 'dir')
    mkdir('results'); % mkdir: creates new folder
end
if ~exist('results/models', 'dir')
    mkdir('results/models');
end

%% Load extracted features
fprintf('Loading extracted features...\n');
load('data/feature-extracted/features.mat');
% This loads all feature sets and labels we created earlier

fprintf('Features loaded successfully.\n');
fprintf('Training samples: %d\n', size(features_raw_train, 1));
fprintf('Validation samples: %d\n\n', size(features_raw_val, 1));

%% Prepare feature sets for training
% Organize all feature sets into a structure for easy iteration
% This makes our code cleaner - we can loop through all features

fprintf('Preparing feature sets...\n');

% Create a cell array to store feature set names
feature_names = {'Raw', 'PCA50', 'PCA100', 'HOG', 'Hybrid'};

% Create structure arrays for training and validation features
% Structure: like a container that holds related data together
features_train = struct();
features_val = struct();

% Assign each feature set to the structure
features_train.Raw = features_raw_train;       % 784 features
features_train.PCA50 = features_pca50_train;   % 50 features
features_train.PCA100 = features_pca100_train; % 100 features
features_train.HOG = features_hog_train;       % ~441 features
features_train.Hybrid = features_hybrid_train; % 50 features

features_val.Raw = features_raw_val;
features_val.PCA50 = features_pca50_val;
features_val.PCA100 = features_pca100_val;
features_val.HOG = features_hog_val;
features_val.Hybrid = features_hybrid_val;

fprintf('Feature sets prepared:\n');
for i = 1:length(feature_names)
    fname = feature_names{i};
    fprintf('  %s: %d features\n', fname, size(features_train.(fname), 2));
    % .(fname) - dot notation to access structure field by name
end
fprintf('\n');

%% Initialize results storage
% Create a table to store validation accuracies for comparison
% rows = feature sets, columns = models

results_table = zeros(length(feature_names), 3);
% 5 rows (feature sets) × 3 columns (models)
% Will store validation accuracy for each combination

model_names = {'k-NN', 'SVM', 'Random Forest'};

%% ========================================
%% MODEL 1: k-Nearest Neighbors (k-NN)
%% ========================================
% How it works: Finds k closest training examples, uses majority vote
% Pros: Simple, no training needed, works well for MNIST
% Cons: Slow prediction, sensitive to k value

fprintf('========================================\n');
fprintf('MODEL 1: k-Nearest Neighbors (k-NN)\n');
fprintf('========================================\n\n');

% We'll test different k values and pick the best one
k_values = [3, 5, 7, 10]; % Common choices for k

for f = 1:length(feature_names)
    fname = feature_names{f};

    fprintf('Training k-NN on %s features...\n', fname);

    % Get current feature set
    X_train = features_train.(fname);
    X_val = features_val.(fname);

    % Hyperparameter tuning: try different k values
    best_k = 3;
    best_acc = 0;

    fprintf('  Tuning k parameter (trying k = %s)...\n', mat2str(k_values));

    for k = k_values
        % Train k-NN classifier
        % fitcknn: MATLAB function to create k-NN model
        % 'NumNeighbors', k: sets how many neighbors to consider
        knn_model = fitcknn(X_train, train_labels, ...
            'NumNeighbors', k, ...
            'Distance', 'euclidean', ... % euclidean: straight-line distance
            'Standardize', false); % already normalized, don't standardize again

        % Predict on validation set
        val_predictions = predict(knn_model, X_val);
        % predict: uses trained model to classify new data

        % Calculate accuracy
        correct = sum(val_predictions == val_labels); % count correct predictions
        accuracy = correct / length(val_labels); % accuracy = correct / total

        fprintf('    k=%d: Validation accuracy = %.2f%%\n', k, accuracy*100);

        % Keep track of best k value
        if accuracy > best_acc
            best_acc = accuracy;
            best_k = k;
        end
    end

    fprintf('  ✓ Best k = %d with accuracy = %.2f%%\n', best_k, best_acc*100);

    % Train final model with best k
    knn_final = fitcknn(X_train, train_labels, ...
        'NumNeighbors', best_k, ...
        'Distance', 'euclidean', ...
        'Standardize', false);

    % Save trained model
    model_filename = sprintf('results/models/knn_%s.mat', lower(fname));
    save(model_filename, 'knn_final', 'best_k', 'best_acc');
    % sprintf: creates formatted string (like printf but returns string)
    % lower: converts to lowercase for consistent filenames

    % Store result in table
    results_table(f, 1) = best_acc;

    fprintf('  Model saved to %s\n\n', model_filename);
end

fprintf('k-NN training complete!\n\n');

%% ========================================
%% MODEL 2: Support Vector Machine (SVM)
%% ========================================
% How it works: Finds optimal hyperplane that separates classes
% Pros: Very accurate, works well in high dimensions
% Cons: Slower to train, sensitive to parameters

fprintf('========================================\n');
fprintf('MODEL 2: Support Vector Machine (SVM)\n');
fprintf('========================================\n\n');

% SVM parameters to try
% Kernel: determines decision boundary shape
% BoxConstraint (C): controls trade-off between margin and errors

fprintf('Note: SVM training may take several minutes per feature set...\n\n');

for f = 1:length(feature_names)
    fname = feature_names{f};

    fprintf('Training SVM on %s features...\n', fname);

    % Get current feature set
    X_train = features_train.(fname);
    X_val = features_val.(fname);

    % For MNIST, linear kernel usually works well and is faster
    % Could also try 'rbf' (radial basis function) for non-linear boundaries

    fprintf('  Using linear kernel...\n');

    % Train SVM using one-vs-all strategy for multiclass (10 digits)
    % fitcecoc: fits Error-Correcting Output Codes model
    % ECOC: strategy for multiclass classification using binary classifiers
    tic; % tic-toc: measures elapsed time
    svm_model = fitcecoc(X_train, train_labels, ...
        'Learners', templateSVM('KernelFunction', 'linear', ...
        'BoxConstraint', 1), ...
        'Coding', 'onevsall');
    % 'onevsall': trains 10 binary classifiers (digit 0 vs rest, 1 vs rest, etc.)
    training_time = toc;

    fprintf('  Training time: %.1f seconds\n', training_time);

    % Predict on validation set
    fprintf('  Evaluating on validation set...\n');
    val_predictions = predict(svm_model, X_val);

    % Calculate accuracy
    correct = sum(val_predictions == val_labels);
    accuracy = correct / length(val_labels);

    fprintf('  ✓ Validation accuracy = %.2f%%\n', accuracy*100);

    % Save trained model
    model_filename = sprintf('results/models/svm_%s.mat', lower(fname));
    save(model_filename, 'svm_model', 'accuracy', 'training_time');

    % Store result in table
    results_table(f, 2) = accuracy;

    fprintf('  Model saved to %s\n\n', model_filename);
end

fprintf('SVM training complete!\n\n');

%% ========================================
%% MODEL 3: Random Forest
%% ========================================
% How it works: Ensemble of decision trees, uses voting
% Pros: Handles high dimensions, robust, provides feature importance
% Cons: Can be slow for large datasets

fprintf('========================================\n');
fprintf('MODEL 3: Random Forest\n');
fprintf('========================================\n\n');

% Test different numbers of trees
num_trees_list = [50, 100]; % More trees = better but slower

for f = 1:length(feature_names)
    fname = feature_names{f};

    fprintf('Training Random Forest on %s features...\n', fname);

    % Get current feature set
    X_train = features_train.(fname);
    X_val = features_val.(fname);

    best_trees = 50;
    best_acc = 0;

    fprintf('  Tuning number of trees (trying %s)...\n', mat2str(num_trees_list));

    for num_trees = num_trees_list
        % Train Random Forest
        % TreeBagger: creates bootstrap-aggregated decision tree ensemble
        % Bootstrap: random sampling with replacement
        rf_model = TreeBagger(num_trees, X_train, train_labels, ...
            'Method', 'classification', ... % classification task (not regression)
            'OOBPrediction', 'on', ... % Out-of-bag: uses left-out samples for validation
            'MinLeafSize', 1); % minimum samples needed in leaf node
        % MinLeafSize=1: allows tree to grow fully (more detailed decisions)

        % Predict on validation set
        % predict returns cell array, convert to double
        val_predictions_cell = predict(rf_model, X_val);
        val_predictions = str2double(val_predictions_cell);
        % str2double: converts string predictions to numbers

        % Calculate accuracy
        correct = sum(val_predictions == val_labels);
        accuracy = correct / length(val_labels);

        fprintf('    %d trees: Validation accuracy = %.2f%%\n', num_trees, accuracy*100);

        if accuracy > best_acc
            best_acc = accuracy;
            best_trees = num_trees;
        end
    end

    fprintf('  ✓ Best: %d trees with accuracy = %.2f%%\n', best_trees, best_acc*100);

    % Train final model with best number of trees
    rf_final = TreeBagger(best_trees, X_train, train_labels, ...
        'Method', 'classification', ...
        'OOBPrediction', 'on', ...
        'MinLeafSize', 1);

    % Save trained model
    model_filename = sprintf('results/models/rf_%s.mat', lower(fname));
    save(model_filename, 'rf_final', 'best_trees', 'best_acc');

    % Store result in table
    results_table(f, 3) = best_acc;

    fprintf('  Model saved to %s\n\n', model_filename);
end

fprintf('Random Forest training complete!\n\n');

%% ========================================
%% Results Summary
%% ========================================

fprintf('========================================\n');
fprintf('TRAINING RESULTS SUMMARY\n');
fprintf('========================================\n\n');

% Display results table
fprintf('Validation Accuracies (in %%):\n\n');
fprintf('%-15s | %-10s | %-10s | %-15s\n', 'Feature Set', 'k-NN', 'SVM', 'Random Forest');
fprintf('----------------+------------+------------+-----------------\n');

for f = 1:length(feature_names)
    fprintf('%-15s | %9.2f%% | %9.2f%% | %14.2f%%\n', ...
        feature_names{f}, ...
        results_table(f, 1)*100, ...
        results_table(f, 2)*100, ...
        results_table(f, 3)*100);
end

fprintf('\n');

%% Find best combination
[max_acc, max_idx] = max(results_table(:)); % max: finds maximum value and its position
% Convert linear index to row, column
[best_feature_idx, best_model_idx] = ind2sub(size(results_table), max_idx);
% ind2sub: converts linear index to subscripts (row, column)

fprintf('🏆 Best Performance:\n');
fprintf('   Model: %s\n', model_names{best_model_idx});
fprintf('   Features: %s\n', feature_names{best_feature_idx});
fprintf('   Validation Accuracy: %.2f%%\n\n', max_acc*100);

%% Visualize results
fprintf('Creating visualization...\n');

figure('Name', 'Model Comparison', 'Position', [100, 100, 1000, 600]);

% Create grouped bar chart
bar(results_table * 100); % multiply by 100 to show as percentage
% bar: creates bar chart, each group = one feature set

% Customize plot
set(gca, 'XTickLabel', feature_names); % gca: get current axes
% XTickLabel: sets labels for x-axis
xlabel('Feature Set');
ylabel('Validation Accuracy (%)');
title('Model Performance Comparison Across Feature Sets');
legend(model_names, 'Location', 'southeast');
grid on;
ylim([90 100]); % ylim: sets y-axis limits (zoom in on 90-100% range)

% Add value labels on bars (optional but nice)
% This shows exact accuracy on each bar
text_offset = 0.5; % small offset above bars
for i = 1:size(results_table, 1)
    for j = 1:size(results_table, 2)
        text(i + (j-2)*0.25, results_table(i,j)*100 + text_offset, ...
            sprintf('%.1f', results_table(i,j)*100), ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 8);
        % text: adds text annotation at specified position
    end
end

%% Save results
fprintf('Saving training results...\n');

save('results/training_results.mat', ...
    'results_table', 'feature_names', 'model_names', ...
    'best_feature_idx', 'best_model_idx', 'max_acc');

fprintf('Results saved to results/training_results.mat\n');

%% Final Summary
fprintf('\n========================================\n');
fprintf('TRAINING COMPLETE!\n');
fprintf('========================================\n\n');

fprintf('Summary:\n');
fprintf('• Trained 15 models (3 algorithms × 5 feature sets)\n');
fprintf('• Best model: %s with %s features\n', ...
    model_names{best_model_idx}, feature_names{best_feature_idx});
fprintf('• All models saved in results/models/\n');
fprintf('• Next step: Detailed evaluation on test set\n\n');

fprintf('Time to move to evaluation and metrics calculation!\n');