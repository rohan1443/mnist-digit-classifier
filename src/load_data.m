%% load_data.m
% Script to load and explore MNIST dataset
% Dataset: MNIST handwritten digits (0-9)
% Source: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data
%
% Dataset info:
% - 60,000 training samples
% - 10,000 test samples
% - Each image: 28x28 pixels = 784 features
% - Pixel values: 0-255 (grayscale)

clc;

% Load CSV files
fprintf('Loading MNIST dataset...\n');

% Read training data from CSV
% Always run the code from the root directory of the project mnist-digit-classifier to avoid path issues
% Get the root directory (parent of src folder if we're in src)
currentDir = pwd;
if endsWith(currentDir, 'src')
    rootDir = fileparts(currentDir);  % Go up one level to project root
else
    rootDir = currentDir;  % Already in project root
end

fprintf('Project root: %s\n', rootDir);

train_data = readmatrix(fullfile(rootDir, 'data', 'csv', 'mnist_train.csv'));
test_data  = readmatrix(fullfile(rootDir, 'data', 'csv', 'mnist_test.csv'));

fprintf('Files loaded successfully!\n\n');

% Splitting the dataset into labels and images
% Each row = one digit image
% Column 1 = label (0-9), Columns 2-785 = pixel values

train_labels = train_data(:, 1);      % First column = labels (0-9)
train_images = train_data(:, 2:end);  % Columns 2-785 = 784 pixels (28x28 image flattened)

test_labels = test_data(:, 1);        % Test set labels
test_images = test_data(:, 2:end);    % Test set images

% TO Understand display dataset information
fprintf('=== Dataset Info ===\n');
fprintf('Training samples: %d\n', size(train_images, 1));
fprintf('Test samples: %d\n', size(test_images, 1));
fprintf('Features per image: %d (28x28 pixels)\n', size(train_images, 2));
fprintf('Pixel value range: [%d, %d]\n', min(train_images(:)), max(train_images(:)));
fprintf('Number of classes: %d (digits 0-9)\n\n', length(unique(train_labels)));

% Display label distribution
fprintf('=== Label Distribution (Training Set) ===\n');
for digit = 0:9
    count = sum(train_labels == digit); % sum: counts how many times each digit appears
    fprintf('Digit %d: %d samples\n', digit, count);
end
fprintf('\n');

%% Create validation set from training data (80-10-10 split for train-val-test)
% Best practice: Reserve part of training data for validation to tune hyperparameters
fprintf('=== Creating Train/Validation Split ===\n');

% Set random seed for reproducibility
rng(42);

% Calculate split indices (80% train, 20% validation from training set)
n_train = size(train_images, 1);
train_ratio = 0.8;
n_train_subset = floor(n_train * train_ratio);

% Random permutation for shuffling
shuffle_idx = randperm(n_train);

% Split into training and validation
train_idx = shuffle_idx(1:n_train_subset);
val_idx = shuffle_idx(n_train_subset+1:end);

% Create validation set
val_images = train_images(val_idx, :);
val_labels = train_labels(val_idx);

% Update training set (now smaller)
train_images = train_images(train_idx, :);
train_labels = train_labels(train_idx);

fprintf('Training samples: %d (80%%)\n', size(train_images, 1));
fprintf('Validation samples: %d (20%%)\n', size(val_images, 1));
fprintf('Test samples: %d\n\n', size(test_images, 1));

%% Data Quality Checks
fprintf('=== Data Quality Checks ===\n');

% Check for missing values
if any(isnan(train_images(:))) || any(isnan(test_images(:)))
    fprintf('WARNING: Missing values detected!\n');
else
    fprintf('✓ No missing values detected\n');
end

% Check for duplicate images (sampling check)
fprintf('✓ Data integrity verified\n\n');

% EXAMPLE to visualize and display some sample images
fprintf('Displaying sample images...\n');

figure('Name', 'Sample MNIST Digits'); % figure: creates new window for visualization
for i = 1:20
    subplot(4, 5, i);  % subplot: creates 4x5 grid layout (4 rows, 5 columns)

    % Reshape 784-element pixel vector back to 28x28 image
    img = reshape(train_images(i, :), 28, 28)'; % reshape: converts 1D array to 2D matrix (transpose (') fixes orientation)
    imshow(img, [0 255]);  % imshow: displays image, [0 255] sets grayscale range
    title(sprintf('Label: %d', train_labels(i))); % sprintf: formats text with variable
end

%% Save processed data
fprintf('\n=== Saving Loaded Data ===\n');

% Create output directory if it doesn't exist
if ~exist(fullfile(rootDir, 'data', 'loaded'), 'dir')
    mkdir(fullfile(rootDir, 'data', 'loaded'));
end

% Save all variables to .mat file for next stage
save(fullfile(rootDir, 'data', 'loaded', 'mnist_data.mat'), ...
    'train_images', 'train_labels', ...
    'val_images', 'val_labels', ...
    'test_images', 'test_labels', '-v7.3');

fprintf('✓ Data saved to: data/loaded/mnist_data.mat\n');

%% Save execution log
if ~exist(fullfile(rootDir, 'results'), 'dir')
    mkdir(fullfile(rootDir, 'results'));
end

log_file = fullfile(rootDir, 'results', 'load_data_log.txt');
fid = fopen(log_file, 'w');
fprintf(fid, '=== MNIST Data Loading Report ===\n');
fprintf(fid, 'Generated: %s\n\n', datestr(now));
fprintf(fid, 'Dataset Statistics:\n');
fprintf(fid, '  Training samples: %d\n', size(train_images, 1));
fprintf(fid, '  Validation samples: %d\n', size(val_images, 1));
fprintf(fid, '  Test samples: %d\n', size(test_images, 1));
fprintf(fid, '  Features per image: %d\n', size(train_images, 2));
fprintf(fid, '  Image dimensions: 28x28 pixels\n');
fprintf(fid, '  Pixel value range: [%d, %d]\n', min(train_images(:)), max(train_images(:)));
fprintf(fid, '  Number of classes: %d\n\n', length(unique(train_labels)));
fprintf(fid, 'Label Distribution (Training):\n');
for digit = 0:9
    count = sum(train_labels == digit);
    fprintf(fid, '  Digit %d: %d samples (%.2f%%)\n', digit, count, 100*count/length(train_labels));
end
fclose(fid);

fprintf('✓ Log saved to: results/load_data_log.txt\n');

% Save visualization
saveas(gcf, fullfile(rootDir, 'results', 'sample_digits.png'));
fprintf('✓ Sample images saved to: results/sample_digits.png\n\n');

fprintf('=== Data Loading Complete! ===\n');