%% evaluate_lda.m - Automatically evaluate best LDA model on test set
% ========== PATHS ==========
dataPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\features';
modelPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\models\lda';
resultsPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\results\lda';

modelFiles = dir(fullfile(modelPath, 'model_lda_*.mat'));
if isempty(modelFiles)
    error('No LDA model found in %s', modelPath);
end

% Auto‑select model with highest validation accuracy (if available)
bestVal = -inf;
bestIdx = 1;
for i = 1:length(modelFiles)
    fileData = load(fullfile(modelPath, modelFiles(i).name));
    if isfield(fileData, 'valAcc')
        val = fileData.valAcc;
    else
        val = -inf; % if missing, treat as worst
    end
    if val > bestVal
        bestVal = val;
        bestIdx = i;
    end
end
modelFile = fullfile(modelPath, modelFiles(bestIdx).name);
if bestVal == -inf
    fprintf('No validation accuracy found in model files; using first file.\n');
else
    fprintf('Auto‑selected LDA model: %s (val acc: %.2f%%)\n', modelFiles(bestIdx).name, bestVal);
end

load(modelFile, 'ldaModel', 'mu_full', 'sigma_full', 'bestFeature');
fprintf('Loaded model trained on %s features.\n', upper(bestFeature));

% Load test features and labels
load(fullfile(dataPath, bestFeature, sprintf('features_%s.mat', bestFeature)), ...
     sprintf('X_test_%s', bestFeature), 'test_labels');
Xtest = eval(sprintf('X_test_%s', bestFeature));

% Standardize using training mu/sigma
Xtest_std = (Xtest - mu_full) ./ sigma_full;

% Predict
predTest = predict(ldaModel, Xtest_std);
testAcc = mean(predTest == categorical(test_labels)) * 100;

fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  LDA TEST RESULTS                                          ║\n');
fprintf('╠════════════════════════════════════════════════════════════╣\n');
fprintf('║  Feature:      %s                                          ║\n', upper(bestFeature));
fprintf('║  Test Accuracy:  %.2f%%                                     ║\n', testAcc);
fprintf('║  Test Samples:   %d                                        ║\n', length(test_labels));
fprintf('╚════════════════════════════════════════════════════════════╝\n');

% Confusion matrix
figure('Name', sprintf('LDA Confusion Matrix - %s', upper(bestFeature)));
cm = confusionchart(categorical(test_labels), predTest);
cm.Title = sprintf('LDA on %s\nTest Accuracy: %.2f%%', upper(bestFeature), testAcc);
xlabel('Predicted Digit'); ylabel('True Digit');
colormap(parula);
set(gcf, 'ToolBar', 'none');
saveas(gcf, fullfile(resultsPath, sprintf('lda_confusion_%s_eval.png', bestFeature)));
fprintf('Confusion matrix saved.\n');