%% train_model.m
% Model Selection (1.6) & Training (1.7)
% Trains: Naive Bayes, k-NN, LDA, Random Forest, SVM.
% Uses PCA features for efficiency; saves models and training report.

function [models, reportPath] = train_model(varargin)
% TRAIN_MODEL  Load features, train classifiers, save models and report.
% Always writes results/reports/training_report.txt.

rootDir = get_root_dir();
ensure_dirs(fullfile(rootDir, 'results', 'models'), fullfile(rootDir, 'results', 'reports'));

featPath = fullfile(rootDir, 'results', 'features', 'features_pca.mat');
if ~exist(featPath, 'file')
    error('Run extract_features.m first.');
end
data = load(featPath);
X_train = data.X_train_pca; train_Y = data.train_Y;
X_val   = data.X_val_pca;   val_Y   = data.val_Y;
X_test  = data.X_test_pca;  test_Y  = data.test_Y;

% For multiclass, use categorical if needed (some fitters expect it)
train_Y = categorical(train_Y);
val_Y   = categorical(val_Y);
test_Y  = categorical(test_Y);

rng(42);
reportLines = {};
reportLines{end+1} = '================================================================================';
reportLines{end+1} = 'TRAINING REPORT - MNIST Handwritten Digit Recognition';
reportLines{end+1} = sprintf('Generated: %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
reportLines{end+1} = '================================================================================';
reportLines{end+1} = '';
reportLines{end+1} = sprintf('Training samples: %d | Validation: %d | Test: %d', size(X_train,1), size(X_val,1), size(X_test,1));
reportLines{end+1} = sprintf('Feature dimension (PCA): %d', size(X_train, 2));
reportLines{end+1} = '';

models = struct();

%% 1. Naive Bayes
reportLines{end+1} = '--- 1. Naive Bayes ---';
tic;
mdl_nb = fitcnb(X_train, train_Y, 'DistributionNames', 'kernel');
models.NaiveBayes = mdl_nb;
t_nb = toc;
pred_val_nb = predict(mdl_nb, X_val);
acc_nb = sum(pred_val_nb == val_Y) / numel(val_Y);
reportLines{end+1} = sprintf('  Validation accuracy: %.4f | Time: %.2f s', acc_nb, t_nb);
reportLines{end+1} = '';

%% 2. k-NN
reportLines{end+1} = '--- 2. k-Nearest Neighbors ---';
k = 5;
tic;
mdl_knn = fitcknn(X_train, train_Y, 'NumNeighbors', k, 'Distance', 'euclidean', 'Standardize', false);
models.kNN = mdl_knn;
t_knn = toc;
pred_val_knn = predict(mdl_knn, X_val);
acc_knn = sum(pred_val_knn == val_Y) / numel(val_Y);
reportLines{end+1} = sprintf('  k = %d, Validation accuracy: %.4f | Time: %.2f s', k, acc_knn, t_knn);
reportLines{end+1} = '';

%% 3. LDA (Linear Discriminant Analysis)
reportLines{end+1} = '--- 3. LDA (Linear Discriminant) ---';
tic;
mdl_lda = fitcdiscr(X_train, train_Y, 'DiscrimType', 'linear');
models.LDA = mdl_lda;
t_lda = toc;
pred_val_lda = predict(mdl_lda, X_val);
acc_lda = sum(pred_val_lda == val_Y) / numel(val_Y);
reportLines{end+1} = sprintf('  Validation accuracy: %.4f | Time: %.2f s', acc_lda, t_lda);
reportLines{end+1} = '';

%% 4. Random Forest (use subset/fast config for report; full run can be done separately)
reportLines{end+1} = '--- 4. Random Forest ---';
nTrees = 100;  % Reduce to 100 for faster run; increase (e.g. 200-500) for best performance
tic;
mdl_rf = TreeBagger(nTrees, X_train, train_Y, 'Method', 'classification', 'MinLeafSize', 10);
models.RandomForest = mdl_rf;
t_rf = toc;
pred_cell = mdl_rf.predict(X_val);
pred_val_rf = categorical(str2double(string(pred_cell)), 0:9);
acc_rf = sum(pred_val_rf == val_Y) / numel(val_Y);
reportLines{end+1} = sprintf('  NumTrees = %d, MinLeafSize = 10 | Validation accuracy: %.4f | Time: %.2f s', nTrees, acc_rf, t_rf);
reportLines{end+1} = '';

%% 5. SVM (multiclass via fitcecoc)
reportLines{end+1} = '--- 5. SVM (one-vs-one) ---';
tic;
mdl_svm = fitcecoc(X_train, train_Y, 'Learners', 'linear', 'Coding', 'onevsone');
models.SVM = mdl_svm;
t_svm = toc;
pred_val_svm = predict(mdl_svm, X_val);
acc_svm = sum(pred_val_svm == val_Y) / numel(val_Y);
reportLines{end+1} = sprintf('  Linear kernel, Validation accuracy: %.4f | Time: %.2f s', acc_svm, t_svm);
reportLines{end+1} = '';

%% Summary and save
reportLines{end+1} = '--- Summary (Validation) ---';
reportLines{end+1} = sprintf('  Naive Bayes:  %.4f', acc_nb);
reportLines{end+1} = sprintf('  k-NN:        %.4f', acc_knn);
reportLines{end+1} = sprintf('  LDA:         %.4f', acc_lda);
reportLines{end+1} = sprintf('  Random Forest: %.4f', acc_rf);
reportLines{end+1} = sprintf('  SVM:         %.4f', acc_svm);

modelDir = fullfile(rootDir, 'results', 'models');
save(fullfile(modelDir, 'trained_models.mat'), 'models');
reportPath = fullfile(rootDir, 'results', 'reports', 'training_report.txt');
fid = fopen(reportPath, 'w');
if fid == -1, error('Cannot create report: %s', reportPath); end
for i = 1:numel(reportLines), fprintf(fid, '%s\n', reportLines{i}); end
fclose(fid);
fprintf('Training report written to: %s\n', reportPath);
end

function rootDir = get_root_dir()
[cwd, name, ~] = fileparts(pwd);
if strcmp(name, 'src'), rootDir = cwd; else, rootDir = pwd; end
end

function ensure_dirs(varargin)
for i = 1:nargin
    if ~exist(varargin{i}, 'dir'), mkdir(varargin{i}); end
end
end
