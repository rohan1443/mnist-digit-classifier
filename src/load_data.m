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
% fprintf('=== Dataset Info ===\n');
% fprintf('Training samples: %d\n', size(train_images, 1));
% fprintf('Test samples: %d\n', size(test_images, 1));
% fprintf('Features per image: %d (28x28 pixels)\n', size(train_images, 2));
% fprintf('Pixel value range: [%d, %d]\n', min(train_images(:)), max(train_images(:)));
% fprintf('Number of classes: %d (digits 0-9)\n\n', length(unique(train_labels)));

% Display label distribution
% fprintf('=== Label Distribution (Training Set) ===\n');
% for digit = 0:9
%     count = sum(train_labels == digit); % sum: counts how many times each digit appears
%     fprintf('Digit %d: %d samples\n', digit, count);
% end
% fprintf('\n');

% % EXAMPLE to visualize and display some sample images
% fprintf('Displaying sample images...\n');

% figure('Name', 'Sample MNIST Digits'); % figure: creates new window for visualization
% for i = 1:20
%     subplot(4, 5, i);  % subplot: creates 4x5 grid layout (4 rows, 5 columns)

%     % Reshape 784-element pixel vector back to 28x28 image
%     img = reshape(train_images(i, :), 28, 28)'; % reshape: converts 1D array to 2D matrix (transpose (') fixes orientation)
%     imshow(img, [0 255]);  % imshow: displays image, [0 255] sets grayscale range
%     title(sprintf('Label: %d', train_labels(i))); % sprintf: formats text with variable
% end

% Aim to split train data into training and validation sets for the model development later

% Set a random seed for reproducibility
rng(42);

% Create a random partition with 20% of the data for validation
% cvpartition: MATLAB function that creates train/validation split
% 'HoldOut', 0.2 means "hold out 20% for validation, use 80% for training"
cv = cvpartition(size(train_images, 1), 'HoldOut', 0.2);

% Get training and validation indices
train_idx = cv.training; % Logical array where true = training samples
val_idx = cv.test;      % Logical array where true = validation samples

% Now i will be extracting the training and validation sets using the indices from the train data bigger sample

train_images_split = train_images(train_idx, :); % Training images (80% of original)
train_labels_split = train_labels(train_idx);    % Training labels (80% of original)

val_images_split = train_images(val_idx, :);     % Validation images (20% of original)
val_labels_split = train_labels(val_idx);        % Validation labels (20% of original)

% renamoing the train set variables for consistency with the new splits
train_images = train_images_split;
train_labels = train_labels_split;

val_images = val_images_split;
val_labels = val_labels_split;

% Verify the sizes of the new datasets to verify the splits
fprintf('Training set size: %d\n', size(train_images_split, 1));
fprintf('Validation set size: %d\n', size(val_images_split, 1));
fprintf('Test set size: %d\n', size(test_images, 1));

% save all the train, validation and the test data in a .mat file for later use in the other stages of the pipeline
save(fullfile(rootDir, 'data', 'loaded', 'mnist_data.mat'), ... % the file locaiton is the loaded folder inside the data folder
    'train_images', 'train_labels', ...
    'val_images', 'val_labels', ...
    'test_images', 'test_labels', '-v7')

fprintf('Data saved to mnist_data.mat successfully!\n')
