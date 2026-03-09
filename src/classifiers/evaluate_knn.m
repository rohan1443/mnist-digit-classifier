%% evaluate_knn.m - Evaluate a trained k-NN model on test set
clc; clear;

% ========== PATHS ==========
dataPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\features';
modelPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\models\knn';
resultsPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\results\knn';

modelFiles = dir(fullfile(modelPath, 'model_knn_*.mat'));
if isempty(modelFiles)
    error('No k-NN model found in %s', modelPath);
end

fprintf('Available k-NN models:\n');
for i = 1:length(modelFiles)
    fprintf('  %d: %s\n', i, modelFiles(i).name);
end
choice = input('Select model number: ');
if isempty(choice) || choice < 1 || choice > length(modelFiles)
    error('Invalid selection.');
end

modelFile = fullfile(modelPath, modelFiles(choice).name);
load(modelFile, 'knnModel', 'mu_full', 'sigma_full', 'bestFeature');

fprintf('\nLoaded model trained on %s features.\n', upper(bestFeature));

% Load test features and labels
load(fullfile(dataPath, bestFeature, sprintf('features_%s.mat', bestFeature)), ...
     sprintf('X_test_%s', bestFeature), 'test_labels');
Xtest = eval(sprintf('X_test_%s', bestFeature));

% Standardize using training mu/sigma
Xtest_std = (Xtest - mu_full) ./ sigma_full;

% Predict
predTest = predict(knnModel, Xtest_std);
testAcc = mean(predTest == categorical(test_labels)) * 100;

fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  K-NN TEST RESULTS                                        ║\n');
fprintf('╠════════════════════════════════════════════════════════════╣\n');
fprintf('║  Feature:      %s                                          ║\n', upper(bestFeature));
fprintf('║  Test Accuracy:  %.2f%%                                     ║\n', testAcc);
fprintf('║  Test Samples:   %d                                        ║\n', length(test_labels));
fprintf('╚════════════════════════════════════════════════════════════╝\n');

% Confusion matrix
figure('Name', sprintf('k-NN Confusion Matrix - %s', upper(bestFeature)));
cm = confusionchart(categorical(test_labels), predTest);
cm.Title = sprintf('k-NN on %s\nTest Accuracy: %.2f%%', upper(bestFeature), testAcc);
xlabel('Predicted Digit'); ylabel('True Digit');
colormap(parula);
set(gcf, 'ToolBar', 'none');
saveas(gcf, fullfile(resultsPath, sprintf('knn_confusion_%s_eval.png', bestFeature)));
fprintf('\nConfusion matrix saved.\n');