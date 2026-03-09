%% train_svm_fast.m - Fast SVM training on HOG (linear kernel)
clc; clear;

dataPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\features';
modelPath = 'C:\Users\T_Jon\mnist-digit-classifier\data\models\svm';

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  FAST SVM TRAINING (LINEAR KERNEL) ON HOG                ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

% Load HOG features
load(fullfile(dataPath, 'hog', 'features_hog.mat'), ...
    'X_train_hog', 'X_val_hog', 'train_labels', 'val_labels');
fprintf('Loaded %d training, %d validation samples.\n', size(X_train_hog,1), size(X_val_hog,1));

% ----- Standardization (still recommended for linear SVM) -----
mu = mean(X_train_hog, 1);
sigma = std(X_train_hog, 0, 1);
sigma(sigma == 0) = 1;
X_train_std = (X_train_hog - mu) ./ sigma;
X_val_std   = (X_val_hog - mu) ./ sigma;

% ----- Train linear SVM -----
fprintf('\nTraining linear SVM...\n');
tic;
svmModel = fitcecoc(X_train_std, categorical(train_labels), ...
    'Learners', templateSVM('KernelFunction', 'linear', 'Standardize', false));
toc;

% Validate
predVal = predict(svmModel, X_val_std);
valAcc = mean(predVal == categorical(val_labels)) * 100;
fprintf('\n✅ Validation Accuracy: %.2f%%\n', valAcc);

% Save
if ~exist(modelPath, 'dir'), mkdir(modelPath); end
save(fullfile(modelPath, 'model_svm_hog_linear.mat'), 'svmModel', 'valAcc', 'mu', 'sigma');
fprintf('Model saved.\n');