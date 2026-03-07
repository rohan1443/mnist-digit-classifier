%% extract_features.m
% Feature Extraction (1.5)
% - Raw pixels and PCA-reduced features for MNIST.
% - Writes results to results/features/ and results/reports/feature_extraction_report.txt

function [X_train_raw, X_val_raw, X_test_raw, X_train_pca, X_val_pca, X_test_pca, reportPath] = extract_features(varargin)
% EXTRACT_FEATURES  Load preprocessed data, compute raw and PCA features.
% Always writes results/reports/feature_extraction_report.txt.

rootDir = get_root_dir();
ensure_dirs(fullfile(rootDir, 'results', 'features'), fullfile(rootDir, 'results', 'reports'));

prepPath = fullfile(rootDir, 'results', 'preprocessed', 'train_val_test.mat');
if ~exist(prepPath, 'file')
    error('Run preprocess_data.m first to create results/preprocessed/train_val_test.mat');
end
data = load(prepPath);
train_X = data.train_X; train_Y = data.train_Y;
val_X   = data.val_X;   val_Y   = data.val_Y;
test_X  = data.test_X; test_Y  = data.test_Y;

%% 1. Raw pixel features (baseline)
% Use normalized 784-dim vector as baseline; suitable for k-NN, SVM, etc.
X_train_raw = train_X;
X_val_raw   = val_X;
X_test_raw  = test_X;
n_raw = size(train_X, 2);

%% 2. PCA for dimensionality reduction
% Justification: 784 features are redundant; PCA captures variance with fewer dimensions,
% reduces overfitting and training time (especially for Random Forest, SVM).
n_components = 100;  % retain top 100 principal components
rng(42);
[coeff, score_train, ~, ~, explained] = pca(train_X, 'NumComponents', n_components);

% Center validation and test using training mean, then project
mu = mean(train_X, 1);
X_train_pca = score_train;
X_val_pca   = (val_X - mu) * coeff;
X_test_pca  = (test_X - mu) * coeff;

variance_retained = sum(explained(1:n_components));

%% 3. Save feature matrices and PCA model for training and Python replication
featDir = fullfile(rootDir, 'results', 'features');
save(fullfile(featDir, 'features_raw.mat'), 'X_train_raw', 'X_val_raw', 'X_test_raw', 'train_Y', 'val_Y', 'test_Y');
save(fullfile(featDir, 'features_pca.mat'), ...
    'X_train_pca', 'X_val_pca', 'X_test_pca', 'train_Y', 'val_Y', 'test_Y', ...
    'coeff', 'mu', 'n_components', 'explained');

%% 4. Feature extraction report
reportPath = fullfile(rootDir, 'results', 'reports', 'feature_extraction_report.txt');
fid = fopen(reportPath, 'w');
if fid == -1, error('Cannot create report: %s', reportPath); end
fprintf(fid, '================================================================================\n');
fprintf(fid, 'FEATURE EXTRACTION REPORT - MNIST Handwritten Digit Recognition\n');
fprintf(fid, 'Generated: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, '================================================================================\n\n');
fprintf(fid, '1. FEATURE EXTRACTION METHODS SELECTED\n');
fprintf(fid, '   (a) Raw pixel features: %d dimensions (28x28 normalized pixels).\n', n_raw);
fprintf(fid, '   (b) PCA: %d components; variance retained = %.2f%%.\n\n', n_components, variance_retained);
fprintf(fid, '2. JUSTIFICATION (WHY THIS METHOD)\n');
fprintf(fid, '   - Raw pixels: Simple baseline; preserves spatial intensity; works well with\n');
fprintf(fid, '     non-linear classifiers (k-NN, SVM, Random Forest).\n');
fprintf(fid, '   - PCA: Reduces dimensionality to avoid curse of dimensionality and speed up\n');
fprintf(fid, '     training (e.g. Random Forest); captures dominant variance; linear method\n');
fprintf(fid, '     suitable for digit images with correlated pixels.\n');
fprintf(fid, '   - Texture/HOG: Not selected for MNIST as digits are thin strokes; raw/PCA\n');
fprintf(fid, '     are standard and sufficient for this dataset.\n\n');
fprintf(fid, '3. OUTPUTS\n');
fprintf(fid, '   - results/features/features_raw.mat  (X_train_raw, X_val_raw, X_test_raw, labels)\n');
fprintf(fid, '   - results/features/features_pca.mat  (X_train_pca, X_val_pca, X_test_pca, coeff, mu, n_components)\n');
fprintf(fid, '   - PCA parameters (coeff, mu) saved for consistent application on new images (e.g. Python deployment).\n');
fclose(fid);
fprintf('Feature extraction report written to: %s\n', reportPath);
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
