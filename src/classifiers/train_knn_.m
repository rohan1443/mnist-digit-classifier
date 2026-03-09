%% train_knn_best.m - Train k-NN on the best feature type
clc; clear;

% ========== PATHS ==========
dataPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\features';
modelPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\models\knn';
resultsPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\results\knn';

if ~exist(modelPath, 'dir'), mkdir(modelPath); end
if ~exist(resultsPath, 'dir'), mkdir(resultsPath); end

featureTypes = {'raw', 'hog', 'pca'};
trainVars = {'X_train_raw', 'X_train_hog', 'X_train_pca'};
testVars  = {'X_test_raw',  'X_test_hog',  'X_test_pca'};

% Load labels
load(fullfile(dataPath, 'raw', 'features_raw.mat'), 'train_labels', 'test_labels');

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  K-NN: FINDING BEST FEATURE ON SUBSET                    ║\n');
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

% Evaluate k-NN on each feature type
bestValAcc = 0;
bestFeature = '';
bestMu = []; bestSigma = [];  % will store standardization params for best feature

fprintf('%-6s : %-10s\n', 'Feature', 'Val Acc');
fprintf('--------------------\n');

for f = 1:length(featureTypes)
    ft = featureTypes{f};
    load(fullfile(dataPath, ft, sprintf('features_%s.mat', ft)), trainVars{f});
    X_full = eval(trainVars{f});
    Xtrain = X_full(subsetIdx(trainSubIdx), :);
    Xval   = X_full(subsetIdx(valSubIdx), :);
    
    % Standardize (kNN needs scaling)
    mu = mean(Xtrain, 1);
    sigma = std(Xtrain, 0, 1);
    sigma(sigma == 0) = 1;
    Xtrain_std = (Xtrain - mu) ./ sigma;
    Xval_std   = (Xval - mu) ./ sigma;
    
    knn_tmp = fitcknn(Xtrain_std, categorical(trainLabels_sub), 'NumNeighbors', 5);
    predVal = predict(knn_tmp, Xval_std);
    valAcc = mean(predVal == categorical(valLabels_sub)) * 100;
    fprintf('%-6s : %6.2f%%\n', upper(ft), valAcc);
    
    if valAcc > bestValAcc
        bestValAcc = valAcc;
        bestFeature = ft;
        bestMu = mu;
        bestSigma = sigma;
    end
end

fprintf('\n✅ Best feature for k-NN: %s (val acc: %.2f%%)\n\n', upper(bestFeature), bestValAcc);

% ========== TRAIN FULL MODEL ==========
fprintf('Training k-NN on full %s data...\n', upper(bestFeature));

load(fullfile(dataPath, bestFeature, sprintf('features_%s.mat', bestFeature)), ...
     trainVars{strcmp(featureTypes, bestFeature)});
Xtrain_full = eval(trainVars{strcmp(featureTypes, bestFeature)});

% Standardize using parameters from subset (or recompute on full – but subset is fine)
% For consistency, we recompute on full training set
mu_full = mean(Xtrain_full, 1);
sigma_full = std(Xtrain_full, 0, 1);
sigma_full(sigma_full == 0) = 1;
Xtrain_std = (Xtrain_full - mu_full) ./ sigma_full;

tic;
knnModel = fitcknn(Xtrain_std, categorical(train_labels), 'NumNeighbors', 5);
trainTime = toc;
fprintf('Training time: %.2f sec\n', trainTime);

% Test
load(fullfile(dataPath, bestFeature, sprintf('features_%s.mat', bestFeature)), ...
     testVars{strcmp(featureTypes, bestFeature)});
Xtest_full = eval(testVars{strcmp(featureTypes, bestFeature)});
Xtest_std = (Xtest_full - mu_full) ./ sigma_full;
predTest = predict(knnModel, Xtest_std);
testAcc = mean(predTest == categorical(test_labels)) * 100;
fprintf('Test accuracy: %.2f%%\n', testAcc);

% Save model
modelFile = fullfile(modelPath, sprintf('model_knn_%s.mat', bestFeature));
save(modelFile, 'knnModel', 'mu_full', 'sigma_full', 'testAcc', 'trainTime', 'bestFeature');

% Save results
resFile = fullfile(resultsPath, sprintf('results_knn_%s.mat', bestFeature));
save(resFile, 'testAcc', 'predTest', 'test_labels', 'bestFeature', 'mu_full', 'sigma_full');

% Confusion matrix
figure('Name', sprintf('k-NN Confusion Matrix - %s', upper(bestFeature)));
cm = confusionchart(categorical(test_labels), predTest);
cm.Title = sprintf('k-NN on %s\nTest Acc: %.2f%%', upper(bestFeature), testAcc);
xlabel('Predicted Digit'); ylabel('True Digit');
colormap(parula);
set(gcf, 'ToolBar', 'none');
saveas(gcf, fullfile(resultsPath, sprintf('knn_confusion_%s.png', bestFeature)));

fprintf('\n✅ k-NN done. Model and results saved.\n');