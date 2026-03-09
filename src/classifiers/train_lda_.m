%% train_lda_best.m - Train LDA on the best feature type (with pseudoLinear)
clc; clear;

% ========== PATHS ==========
dataPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\features';
modelPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\models\lda';
resultsPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\results\lda';

if ~exist(modelPath, 'dir'), mkdir(modelPath); end
if ~exist(resultsPath, 'dir'), mkdir(resultsPath); end

featureTypes = {'raw', 'hog', 'pca'};
trainVars = {'X_train_raw', 'X_train_hog', 'X_train_pca'};
testVars  = {'X_test_raw',  'X_test_hog',  'X_test_pca'};

load(fullfile(dataPath, 'raw', 'features_raw.mat'), 'train_labels', 'test_labels');

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  LDA: FINDING BEST FEATURE ON SUBSET                      ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

subsetSize = 5000;
numSub = subsetSize;
rng(42);
subsetIdx = randperm(length(train_labels), subsetSize);

numVal = round(subsetSize * 0.2);
numTrain = subsetSize - numVal;
trainSubIdx = 1:numTrain;
valSubIdx   = numTrain+1:numSub;

trainLabels_sub = train_labels(subsetIdx(trainSubIdx));
valLabels_sub   = train_labels(subsetIdx(valSubIdx));

fprintf('Subset size: %d (train: %d, val: %d)\n\n', subsetSize, numTrain, numVal);

% Evaluate LDA on each feature type
bestValAcc = 0;
bestFeature = '';
bestMu = []; bestSigma = [];

fprintf('%-6s : %-10s\n', 'Feature', 'Val Acc');
fprintf('--------------------\n');

for f = 1:length(featureTypes)
    ft = featureTypes{f};
    load(fullfile(dataPath, ft, sprintf('features_%s.mat', ft)), trainVars{f});
    X_full = eval(trainVars{f});
    Xtrain = X_full(subsetIdx(trainSubIdx), :);
    Xval   = X_full(subsetIdx(valSubIdx), :);
    
    % Standardize
    mu = mean(Xtrain, 1);
    sigma = std(Xtrain, 0, 1);
    sigma(sigma == 0) = 1;
    Xtrain_std = (Xtrain - mu) ./ sigma;
    Xval_std   = (Xval - mu) ./ sigma;
    
    % Use 'pseudoLinear' to handle singular covariance matrices
    lda_tmp = fitcdiscr(Xtrain_std, categorical(trainLabels_sub), ...
        'DiscrimType', 'pseudoLinear');
    predVal = predict(lda_tmp, Xval_std);
    valAcc = mean(predVal == categorical(valLabels_sub)) * 100;
    fprintf('%-6s : %6.2f%%\n', upper(ft), valAcc);
    
    if valAcc > bestValAcc
        bestValAcc = valAcc;
        bestFeature = ft;
        bestMu = mu;
        bestSigma = sigma;
    end
end

fprintf('\n✅ Best feature for LDA: %s (val acc: %.2f%%)\n\n', upper(bestFeature), bestValAcc);

% ========== TRAIN FULL MODEL ==========
fprintf('Training LDA on full %s data...\n', upper(bestFeature));

load(fullfile(dataPath, bestFeature, sprintf('features_%s.mat', bestFeature)), ...
     trainVars{strcmp(featureTypes, bestFeature)});
Xtrain_full = eval(trainVars{strcmp(featureTypes, bestFeature)});

% Standardize using full training set
mu_full = mean(Xtrain_full, 1);
sigma_full = std(Xtrain_full, 0, 1);
sigma_full(sigma_full == 0) = 1;
Xtrain_std = (Xtrain_full - mu_full) ./ sigma_full;

tic;
% Use 'pseudoLinear' again for robustness
ldaModel = fitcdiscr(Xtrain_std, categorical(train_labels), ...
    'DiscrimType', 'pseudoLinear');
trainTime = toc;
fprintf('Training time: %.2f sec\n', trainTime);

% Test
load(fullfile(dataPath, bestFeature, sprintf('features_%s.mat', bestFeature)), ...
     testVars{strcmp(featureTypes, bestFeature)});
Xtest_full = eval(testVars{strcmp(featureTypes, bestFeature)});
Xtest_std = (Xtest_full - mu_full) ./ sigma_full;
predTest = predict(ldaModel, Xtest_std);
testAcc = mean(predTest == categorical(test_labels)) * 100;
fprintf('Test accuracy: %.2f%%\n', testAcc);

% Save model
modelFile = fullfile(modelPath, sprintf('model_lda_%s.mat', bestFeature));
save(modelFile, 'ldaModel', 'mu_full', 'sigma_full', 'testAcc', 'trainTime', 'bestFeature');

% Save results
resFile = fullfile(resultsPath, sprintf('results_lda_%s.mat', bestFeature));
save(resFile, 'testAcc', 'predTest', 'test_labels', 'bestFeature', 'mu_full', 'sigma_full');

% Confusion matrix
figure('Name', sprintf('LDA Confusion Matrix - %s', upper(bestFeature)));
cm = confusionchart(categorical(test_labels), predTest);
cm.Title = sprintf('LDA on %s\nTest Acc: %.2f%%', upper(bestFeature), testAcc);
xlabel('Predicted Digit'); ylabel('True Digit');
colormap(parula);
set(gcf, 'ToolBar', 'none');
saveas(gcf, fullfile(resultsPath, sprintf('lda_confusion_%s.png', bestFeature)));

fprintf('\n✅ LDA done. Model and results saved.\n');