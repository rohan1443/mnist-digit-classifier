%% preprocess_data.m
% Dataset Selection and Preparation (1.3) & Pre-processing (1.4)
% - Load MNIST data, clean, normalize, split into train/validation/test.
% - Writes results to results/preprocessed/ and results/reports/preprocessing_report.txt

function [train_X, train_Y, val_X, val_Y, test_X, test_Y, reportPath] = preprocess_data(varargin)
% PREPROCESS_DATA  Preprocess MNIST and split into train/val/test.
% Optional: pass (train_images, train_labels, test_images, test_labels) to skip loading.
% Always writes results/reports/preprocessing_report.txt.

if nargin >= 4
    train_images = varargin{1}; train_labels = varargin{2};
    test_images = varargin{3};  test_labels  = varargin{4};
else
    rootDir = get_root_dir();
    [train_images, train_labels, test_images, test_labels] = load_mnist_from_csv(rootDir);
end
rootDir = get_root_dir();
ensure_dirs(fullfile(rootDir, 'results', 'preprocessed'), fullfile(rootDir, 'results', 'reports'));

%% 1. Data understanding (for report)
n_train = size(train_images, 1);
n_test  = size(test_images, 1);
n_features_raw = size(train_images, 2);
n_classes = length(unique([train_labels; test_labels]));
class_list = (0:9)';

%% 2. Cleaning (minimal: MNIST is already clean; remove any all-zero rows if desired)
% Keep all samples; MNIST has no missing values.

%% 3. Normalization (scale pixel 0-255 to 0-1)
% Justification: improves convergence for many classifiers and avoids dominance by scale.
minPix = 0;
maxPix = 255;
train_X = double(train_images);
test_X  = double(test_images);
train_X = (train_X - minPix) / (maxPix - minPix);
test_X  = (test_X - minPix) / (maxPix - minPix);
train_Y = train_labels;
test_Y  = test_labels;

%% 4. Split training into train + validation (e.g. 85% train, 15% validation)
rng(42);
n_val = round(0.15 * n_train);
idx = randperm(n_train);
val_idx = idx(1:n_val);
tr_idx  = idx(n_val+1:end);

val_X = train_X(val_idx, :);
val_Y = train_Y(val_idx);
train_X = train_X(tr_idx, :);
train_Y = train_Y(tr_idx);
n_train_final = size(train_X, 1);
n_val_final   = size(val_X, 1);

%% 5. Save preprocessed data for downstream steps
prepDir = fullfile(rootDir, 'results', 'preprocessed');
save(fullfile(prepDir, 'train_val_test.mat'), ...
    'train_X', 'train_Y', 'val_X', 'val_Y', 'test_X', 'test_Y', ...
    'n_train_final', 'n_val_final', 'n_test', 'n_features_raw', 'n_classes', 'class_list');

%% 6. Write preprocessing report
reportPath = fullfile(rootDir, 'results', 'reports', 'preprocessing_report.txt');
fid = fopen(reportPath, 'w');
if fid == -1, error('Cannot create report file: %s', reportPath); end
fprintf(fid, '================================================================================\n');
fprintf(fid, 'PREPROCESSING REPORT - MNIST Handwritten Digit Recognition\n');
fprintf(fid, 'Generated: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, '================================================================================\n\n');
fprintf(fid, '1. INPUTS / DATA UNDERSTANDING\n');
fprintf(fid, '   - Dataset: MNIST (Kaggle CSV)\n');
fprintf(fid, '   - Original training samples: %d\n', n_train);
fprintf(fid, '   - Original test samples: %d\n', n_test);
fprintf(fid, '   - Raw features per image: %d (28x28 pixels)\n', n_features_raw);
fprintf(fid, '   - Pixel value range (original): [0, 255]\n');
fprintf(fid, '   - Number of classes: %d (digits 0-9)\n\n', n_classes);
fprintf(fid, '2. PREPROCESSING JUSTIFICATIONS\n');
fprintf(fid, '   - Number of samples after split:\n');
fprintf(fid, '     Training set:   %d\n', n_train_final);
fprintf(fid, '     Validation set: %d\n', n_val_final);
fprintf(fid, '     Test set:       %d\n', n_test);
fprintf(fid, '   - Number of features (before feature extraction): %d\n', n_features_raw);
fprintf(fid, '   - Number of classes: %d\n', n_classes);
fprintf(fid, '   - Normalization: min-max scaling to [0, 1] for stability and classifier performance.\n');
fprintf(fid, '   - Train/validation split: 85%% / 15%% (random, seed=42) for model selection.\n\n');
fprintf(fid, '3. CHALLENGES AND HANDLING\n');
fprintf(fid, '   - Challenge: High dimensionality (784 features). Handled in feature extraction (PCA/dimensionality reduction).\n');
fprintf(fid, '   - Challenge: Class balance. MNIST is roughly balanced; no resampling applied.\n');
fprintf(fid, '   - Reproducibility: Fixed rng(42) for train/val split.\n\n');
fprintf(fid, '4. OUTPUTS\n');
fprintf(fid, '   - Saved file: results/preprocessed/train_val_test.mat\n');
fprintf(fid, '   - Variables: train_X, train_Y, val_X, val_Y, test_X, test_Y\n');
fprintf(fid, '   - Normalized pixel range: [0, 1]\n');
fclose(fid);
fprintf('Preprocessing report written to: %s\n', reportPath);
end

function [train_images, train_labels, test_images, test_labels] = load_mnist_from_csv(rootDir)
train_data = readmatrix(fullfile(rootDir, 'data', 'csv', 'mnist_train.csv'));
test_data  = readmatrix(fullfile(rootDir, 'data', 'csv', 'mnist_test.csv'));
train_labels = train_data(:, 1);
train_images = train_data(:, 2:end);
test_labels  = test_data(:, 1);
test_images  = test_data(:, 2:end);
end

function rootDir = get_root_dir()
% Assume we are run from project root or src/; find project root.
[cwd, name, ext] = fileparts(pwd);
if strcmp(name, 'src')
    rootDir = cwd;
else
    rootDir = pwd;
end
end

function ensure_dirs(varargin)
for i = 1:nargin
    d = varargin{i};
    if ~exist(d, 'dir'), mkdir(d); end
end
end
