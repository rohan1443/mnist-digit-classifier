%% train_nb_best.m - Train Naive Bayes on the best feature type
clc; clear;

% ========== PATHS ==========
dataPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\features';
modelPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\models\nb';
resultsPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\results\nb';

if ~exist(modelPath, 'dir'), mkdir(modelPath); end
if ~exist(resultsPath, 'dir'), mkdir(resultsPath); end

% Feature types to consider
featureTypes = {'raw', 'hog', 'pca'};
trainVars = {'X_train_raw', 'X_train_hog', 'X_train_pca'};
testVars  = {'X_test_raw',  'X_test_hog',  'X_test_pca'};

% ========== LOAD LABELS (they are the same for all features) ==========
load(fullfile(dataPath, 'raw', 'features_raw.mat'), 'train_labels', 'test_labels');

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  NAIVE BAYES: FINDING BEST FEATURE ON SUBSET             ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

% Use a random subset of training data for quick comparison
subsetSize = 5000;  % adjust if needed (e.g., 3000)
rng(42); % for reproducibility
subsetIdx = randperm(length(train_labels), subsetSize);

% Split subset into train/validation (80/20)
numSub = subsetSize;
numVal = round(numSub * 0.2);
numTrain = numSub - numVal;
trainSubIdx = 1:numTrain;
valSubIdx   = numTrain+1:numSub;

trainLabels_sub = train_labels(subsetIdx(trainSubIdx));
valLabels_sub   = train_labels(subsetIdx(valSubIdx));

fprintf('Subset size: %d (train: %d, val: %d)\n\n', subsetSize, numTrain, numVal);

% Evaluate NB on each feature type
bestValAcc = 0;
bestFeature = '';

fprintf('%-6s : %-10s\n', 'Feature', 'Val Acc');
fprintf('--------------------\n');

for f = 1:length(featureTypes)
    ft = featureTypes{f};
    % Load training subset for this feature
    load(fullfile(dataPath, ft, sprintf('features_%s.mat', ft)), trainVars{f});
    X_full = eval(trainVars{f});
    Xtrain = X_full(subsetIdx(trainSubIdx), :);
    Xval   = X_full(subsetIdx(valSubIdx), :);
    
    % Train NB (no scaling needed)
    nbModel_tmp = fitcnb(Xtrain, categorical(trainLabels_sub), 'DistributionNames', 'normal');
    predVal = predict(nbModel_tmp, Xval);
    valAcc = mean(predVal == categorical(valLabels_sub)) * 100;
    fprintf('%-6s : %6.2f%%\n', upper(ft), valAcc);
    
    if valAcc > bestValAcc
        bestValAcc = valAcc;
        bestFeature = ft;
    end
end

fprintf('\n✅ Best feature for Naive Bayes: %s (val acc: %.2f%%)\n\n', upper(bestFeature), bestValAcc);

% ========== NOW TRAIN ON FULL DATA WITH BEST FEATURE ==========
fprintf('Training Naive Bayes on full %s data...\n', upper(bestFeature));

% Load full training data for best feature
load(fullfile(dataPath, bestFeature, sprintf('features_%s.mat', bestFeature)), ...
     trainVars{strcmp(featureTypes, bestFeature)});
Xtrain_full = eval(trainVars{strcmp(featureTypes, bestFeature)});

tic;
nbModel = fitcnb(Xtrain_full, categorical(train_labels), 'DistributionNames', 'normal');
trainTime = toc;
fprintf('Training time: %.2f sec\n', trainTime);

% Evaluate on test set
load(fullfile(dataPath, bestFeature, sprintf('features_%s.mat', bestFeature)), ...
     testVars{strcmp(featureTypes, bestFeature)});
Xtest_full = eval(testVars{strcmp(featureTypes, bestFeature)});
predTest = predict(nbModel, Xtest_full);
testAcc = mean(predTest == categorical(test_labels)) * 100;
fprintf('Test accuracy: %.2f%%\n', testAcc);

% Save model
modelFile = fullfile(modelPath, sprintf('model_nb_%s.mat', bestFeature));
save(modelFile, 'nbModel', 'testAcc', 'trainTime', 'bestFeature');

% Save results
resFile = fullfile(resultsPath, sprintf('results_nb_%s.mat', bestFeature));
save(resFile, 'testAcc', 'predTest', 'test_labels', 'bestFeature');

% Generate confusion matrix
figure('Name', sprintf('NB Confusion Matrix - %s', upper(bestFeature)));
cm = confusionchart(categorical(test_labels), predTest);
cm.Title = sprintf('Naive Bayes on %s\nTest Acc: %.2f%%', upper(bestFeature), testAcc);
xlabel('Predicted Digit'); ylabel('True Digit');
colormap(parula);
set(gcf, 'ToolBar', 'none');  % <-- add this line
saveas(gcf, fullfile(resultsPath, sprintf('nb_confusion_%s.png', bestFeature)));

fprintf('\n✅ Naive Bayes done. Model and results saved in:\n  %s\n', modelPath);