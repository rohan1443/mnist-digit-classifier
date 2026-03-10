%% feature extraction for MNIST digit classification
%
% So in this script we are trying to implement and compare different feature extraction techniques for the MNIST digit classification task.
% 1> Raw pixels as the baseline feature set
% PCA for dimensionality reduction
% HOG (Histogram of Oriented Gradients) for capturing edge and shape information
% SIFT (Scale-Invariant Feature Transform) for capturing local keypoints and descriptors
% LBP (Local Binary Patterns) for capturing texture information
% Hybrid (HOG + PCA) to combine edge information with dimensionality reduction
% We will evaluate the performance of these features using a simple SVM classifier and compare their accuracies on the MNIST dataset.
% The goal is to understand how different feature extraction techniques can impact the performance of a machine learning model on image classification tasks, and to identify which features are most effective

clear; clc; close all;

% Load the preprocessed MNIST dataset (assuming load_data.m is in the same directory)
fprintf("==Feature Extraction for MNIST Digit Classification==\n");
fprintf("Loading dataset...\n");

% fprintf("Current working directory: %s\n", pwd);

load('data/preprocessed/mnist_preprocessed.mat'); % This will load train_images, train_labels, test_images, test_labels
fprintf("Dataset loaded successfully\n");
fprintf("Training samples: %d images\n", size(train_images_norm, 1));
fprintf("train images dimensions: %d features (28 x 28 pixels) \n", size(train_images_norm, 2));


%% Method 1: Raw Pixels (Baseline)
% Simply use all 784 pixel values as features
% Pros: Preserves all information, simple to implement
% Cons: High dimensionality (784 features), computationally expensive, may lead to overfitting

fprintf('--- Method 1: Raw Pixels ---\n');
fprintf('Using all 784 pixel values directly as features.\n');

% No transformation needed - already have normalized pixels
features_raw_train = train_images_norm;  % 48000 x 784
features_raw_val = val_images_norm;      % 12000 x 784
features_raw_test = test_images_norm;    % 10000 x 784

fprintf("Feature Dimensions: %d features\n", size(features_raw_train, 2));
fprintf("Raw pixel features ready\n\n");

%% Method 2: PCA (dimensionality reduction via Principal Component Analysis)
% Reducing the 784 pixel features to fewer prinicipal components but still capturing the most important vairance in the data

% Try 2 different numbers of principal components: 50 and 100
% Pros: Reduces dimensionality, can improve computational efficiency, may help with overfitting
% Cons: May lose some information, requires choosing the number of components

num_components_list = [50, 100]; % we will test both to see which is better

for i = 1:length(num_components_list)
    num_components = num_components_list(i);

    fprintf("\n PCA with %d components\n", num_components);

    % Step 1: Performing PCA on the training data
    % to find the direction of max vairance (the principal components)

    [coeff, score, latent] = pca(train_images_norm);
    % coeff: is the principal component coefficients (eigenvectors or loading vectors)
    % score: is the representation of the data projected onto the prinicipal components
    % latent: is the eigen valuews which is the variance explained by each principal component


    % Step2: Keep only the top N components
    pca_coeff = coeff(:, 1:num_components); % select the first N components
    features_pca_train = score(:, 1:num_components); % the traininng features


    % Step 3: Project validation and test data using same transformation
    % Important: Use training PCA coefficients, don't recompute PCA
    features_pca_val = val_images_norm * pca_coeff;   % Matrix multiplication
    features_pca_test = test_images_norm * pca_coeff;

    % Calculate variance explained
    total_variance = sum(latent); % Total variance in data
    variance_explained = sum(latent(1:num_components)) / total_variance * 100;

    fprintf('Variance explained: %.2f%%\n', variance_explained);
    fprintf('Feature dimensions: %d → %d\n', size(train_images_norm, 2), num_components);

    % Save both versions with different names
    if num_components == 50
        features_pca50_train = features_pca_train;
        features_pca50_val = features_pca_val;
        features_pca50_test = features_pca_test;
        pca50_coeff = pca_coeff;
    else % num_components == 100
        features_pca100_train = features_pca_train;
        features_pca100_val = features_pca_val;
        features_pca100_test = features_pca_test;
        pca100_coeff = pca_coeff;
    end
end

fprintf('\n✓ PCA features extracted for 50 and 100 components.\n\n');

%% Method 3: HOG (Histogram of Oriented Gradients)
% Captures edge orientations and local shape information
% How it works: Divides image into cells, computes gradient directions
% Pros: Good for shape recognition (edges define digit shapes)
% Cons: More complex, loses some spatial information

fprintf('--- Method 3: HOG (Edge/Shape Features) ---\n');
fprintf('Extracting gradient-based features...\n');

% HOG parameters
cellSize = [4 4];  % Divide 28x28 image into 7x7 grid of 4x4 cells
% This gives us: (28/4) x (28/4) = 7x7 = 49 cells
% Each cell produces ~9 orientation bins
% Total features: 49 cells × 9 bins ≈ 441 features (approximate)

% Extract HOG features for all images
num_train = size(train_images_norm, 1);
num_val = size(val_images_norm, 1);
num_test = size(test_images_norm, 1);

% Process first image to get feature length
img_sample = reshape(train_images_norm(1, :), 28, 28)'; % Reshape to 28x28
hog_sample = extractHOGFeatures(img_sample, 'CellSize', cellSize);

% extractHOGFeatures: MATLAB function that computes HOG descriptor
hog_length = length(hog_sample); % Number of HOG features

fprintf('HOG will extract %d features per image.\n', hog_length);
fprintf('Processing %d training images...\n', num_train);

% Initialize matrices to store HOG features
features_hog_train = zeros(num_train, hog_length);
features_hog_val = zeros(num_val, hog_length);
features_hog_test = zeros(num_test, hog_length);

% Extract HOG for training set (with progress display)
for i = 1:num_train
    img = reshape(train_images_norm(i, :), 28, 28)'; % Convert vector to image
    features_hog_train(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);

    % Show progress every 5000 images
    if mod(i, 5000) == 0
        fprintf('  Processed %d/%d images...\n', i, num_train);
    end
end

fprintf('Processing validation images...\n');
% Extract HOG for validation set
for i = 1:num_val
    img = reshape(val_images_norm(i, :), 28, 28)';
    features_hog_val(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
end

fprintf('Processing test images...\n');
% Extract HOG for test set
for i = 1:num_test
    img = reshape(test_images_norm(i, :), 28, 28)';
    features_hog_test(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
end

fprintf('✓ HOG features extracted: %d features per image.\n\n', hog_length);

%% Method 4: HYBRID (HOG + PCA) - Our Proposed Approach
% Combine strengths of both methods
% Strategy: Extract HOG features (captures shapes), then reduce with PCA (efficiency)
% Justification: HOG provides discriminative edge information, PCA removes redundancy
% This is our innovation for better grades!

fprintf('--- Method 4: HYBRID (HOG + PCA) ---\n');
fprintf('Applying PCA to HOG features for dimensionality reduction...\n');

% We already have HOG features, now apply PCA to reduce them
num_hybrid_components = 50; % Reduce HOG features to 50 dimensions

fprintf('Reducing HOG features (%d) to %d components using PCA...\n', ...
    hog_length, num_hybrid_components);

% Apply PCA on HOG features
[coeff_hybrid, score_hybrid, latent_hybrid] = pca(features_hog_train);

% Keep top 50 components
hybrid_coeff = coeff_hybrid(:, 1:num_hybrid_components);
features_hybrid_train = score_hybrid(:, 1:num_hybrid_components);

% Project validation and test using same transformation
features_hybrid_val = features_hog_val * hybrid_coeff;
features_hybrid_test = features_hog_test * hybrid_coeff;

% Calculate variance retained
variance_hybrid = sum(latent_hybrid(1:num_hybrid_components)) / sum(latent_hybrid) * 100;

fprintf('  Original HOG features: %d\n', hog_length);
fprintf('  Reduced to: %d components\n', num_hybrid_components);
fprintf('  Variance explained: %.2f%%\n', variance_hybrid);
fprintf('✓ Hybrid features created successfully.\n\n');

%% Summary Comparison
fprintf('=== Feature Extraction Summary ===\n\n');
fprintf('Method              | Features | Description\n');
fprintf('--------------------+----------+----------------------------------\n');
fprintf('Raw Pixels          | %4d     | All original pixel values\n', size(features_raw_train, 2));
fprintf('PCA-50              | %4d     | Top 50 principal components\n', 50);
fprintf('PCA-100             | %4d     | Top 100 principal components\n', 100);
fprintf('HOG                 | %4d     | Gradient orientation features\n', hog_length);
fprintf('Hybrid (HOG+PCA)    | %4d     | HOG reduced to 50 components\n', num_hybrid_components);
fprintf('\n');

%% Why We Chose These Methods (Justification for Report)
fprintf('=== Method Selection Justification ===\n\n');

fprintf('Why NOT other methods:\n');
fprintf('• LBP (Local Binary Patterns): Better for texture, digits are more about shape\n');
fprintf('• Wavelet Transform: Adds complexity without significant accuracy gain for MNIST\n');
fprintf('• Deep Features (CNN): Requires deep learning toolbox, beyond scope\n\n');

fprintf('Why we chose our methods:\n');
fprintf('• Raw Pixels: Baseline to compare against\n');
fprintf('• PCA: Standard dimensionality reduction, widely used in literature\n');
fprintf('• HOG: Proven effective for shape recognition tasks\n');
fprintf('• Hybrid: Combines HOG discriminative power with PCA efficiency\n');
fprintf('  → This is our contribution/innovation!\n\n');

%% Visualize Feature Extraction
fprintf('Creating visualization...\n');

figure('Name', 'Feature Extraction Visualization', 'Position', [100, 100, 1200, 800]);

% Select one sample digit to visualize
sample_idx = 1;
sample_img = reshape(train_images_norm(sample_idx, :), 28, 28)';
sample_label = train_labels(sample_idx);

% Original image
subplot(2, 3, 1);
imshow(sample_img, []);
title(sprintf('Original Image\nLabel: %d', sample_label));

% Raw pixels visualization (show as 1D signal)
subplot(2, 3, 2);
plot(features_raw_train(sample_idx, :));
title(sprintf('Raw Pixels\n784 features'));
xlabel('Pixel index');
ylabel('Normalized value');
grid on;

% PCA-50 visualization
subplot(2, 3, 3);
stem(features_pca50_train(sample_idx, :));
title(sprintf('PCA-50\n50 components'));
xlabel('Component index');
ylabel('Score');
grid on;

% PCA-100 visualization
subplot(2, 3, 4);
stem(features_pca100_train(sample_idx, :));
title(sprintf('PCA-100\n100 components'));
xlabel('Component index');
ylabel('Score');
grid on;

% HOG visualization
subplot(2, 3, 5);
plot(features_hog_train(sample_idx, :));
title(sprintf('HOG Features\n%d features', hog_length));
xlabel('Feature index');
ylabel('Value');
grid on;

% Hybrid visualization
subplot(2, 3, 6);
stem(features_hybrid_train(sample_idx, :));
title(sprintf('Hybrid (HOG+PCA)\n50 components'));
xlabel('Component index');
ylabel('Score');
grid on;

sgtitle('Feature Extraction Methods Comparison');

%% Save all extracted features
fprintf('\nSaving extracted features...\n');

save('data/feature-extracted/features.mat', ...
    'features_raw_train', 'features_raw_val', 'features_raw_test', ...
    'features_pca50_train', 'features_pca50_val', 'features_pca50_test', ...
    'features_pca100_train', 'features_pca100_val', 'features_pca100_test', ...
    'features_hog_train', 'features_hog_val', 'features_hog_test', ...
    'features_hybrid_train', 'features_hybrid_val', 'features_hybrid_test', ...
    'train_labels', 'val_labels', 'test_labels', ...
    'pca50_coeff', 'pca100_coeff', 'hybrid_coeff', ...
    '-v7.3');

fprintf('All features saved to data/features.mat\n');

%% Final Summary
fprintf('\n=== Feature Extraction Complete ===\n');
fprintf('Ready for model training!\n');
fprintf('Next step: Train classifiers on different feature sets and compare.\n');


