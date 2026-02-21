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
rootDir = pwd;
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