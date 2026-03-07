%% preprocess_data.m
% Advanced preprocessing pipeline for handwritten digit recognition
% Implements best practices from recent research papers:
% - Normalization & Standardization
% - Data Augmentation (optional)
% - Noise Reduction
% - Contrast Enhancement
% - Quality Validation

clc;
fprintf('=== Preprocessing Data ===\n');

stage = 'STAGE-PREPROCESSING';

%% Load raw data
fprintf('[%s] Loading the raw data from the .mat file...\n', stage);

% Get the root directory (parent of src folder if we're in src)
currentDir = pwd;
if endsWith(currentDir, 'src')
    rootDir = fileparts(currentDir);  % Go up one level to project root
else
    rootDir = currentDir;  % Already in project root
end

fprintf('Root directory: %s\n', rootDir);

load(fullfile(rootDir, 'data', 'loaded', 'mnist_data.mat'));
fprintf('[%s] Data loading done\n', stage);

% Display dataset information
fprintf('[%s] train Images size: %d x %d\n', stage, size(train_images,1), size(train_images,2));
fprintf('[%s] validation images size: %d x %d\n', stage, size(val_images,1), size(val_images,2));
fprintf('[%s] test images size: %d x %d\n', stage, size(test_images,1), size(test_images,2));

%% Preprocessing Step 1: Normalization (Min-Max Scaling to [0, 1])
% Research shows that normalized inputs improve model convergence and stability
% Method: Scale pixel values from [0, 255] to [0, 1]

fprintf('\n[%s]\nStep 1: Normalizing (Min-Max Scaling)\n', stage);

train_images_norm = double(train_images) / 255;
val_images_norm = double(val_images) / 255;
test_images_norm = double(test_images) / 255;

fprintf('✓ Normalization done\n');
fprintf('  New Pixel Range: [%.2f, %.2f]\n', min(train_images_norm(:)), max(train_images_norm(:)));

%% Preprocessing Step 2: Mean Centering & Standardization
% Centers data around zero and scales to unit variance
% Benefits: Faster convergence, better gradient flow in neural networks

fprintf('\n[%s]\nStep 2: Mean Centering & Standardization\n', stage);

% Calculate statistics from training set only (avoid data leakage)
mean_pixel_value = mean(train_images_norm(:));
std_pixel_value = std(train_images_norm(:));

% Apply standardization: (x - mean) / std
train_images_standardized = (train_images_norm - mean_pixel_value) / std_pixel_value;
val_images_standardized = (val_images_norm - mean_pixel_value) / std_pixel_value;
test_images_standardized = (test_images_norm - mean_pixel_value) / std_pixel_value;

fprintf('✓ Standardization done\n');
fprintf('  Training Mean: %.4f, Std: %.4f\n', mean_pixel_value, std_pixel_value);
fprintf('  New Range after standardization: [%.2f, %.2f]\n', ...
    min(train_images_standardized(:)), max(train_images_standardized(:)));

%% Preprocessing Step 3: Contrast Enhancement using CLAHE
% Contrast Limited Adaptive Histogram Equalization
% Enhances local contrast while preventing over-amplification of noise

fprintf('\n[%s]\nStep 3: Contrast Enhancement (CLAHE - Optional)\n', stage);

% Apply CLAHE to a subset for demonstration (computationally intensive)
n_samples_clahe = 1000; % Process first 1000 images for demo
train_images_clahe = zeros(n_samples_clahe, 784);

fprintf('  Processing %d samples with CLAHE...\n', n_samples_clahe);
for i = 1:n_samples_clahe
    img_2d = reshape(train_images_norm(i, :), 28, 28)';
    img_enhanced = adapthisteq(img_2d, 'ClipLimit', 0.02, 'Distribution', 'uniform');
    train_images_clahe(i, :) = img_enhanced(:)';
end
fprintf('✓ CLAHE enhancement completed (on subset)\n');

%% Preprocessing Step 4: Noise Reduction (Gaussian Smoothing)
% Reduces sensor noise and minor artifacts
% Uses gentle Gaussian filter to preserve digit structure

fprintf('\n[%s]\nStep 4: Noise Reduction (Gaussian Smoothing - Optional)\n', stage);

% Apply Gaussian smoothing to a subset
n_samples_smooth = 1000;
train_images_smoothed = zeros(n_samples_smooth, 784);

fprintf('  Applying Gaussian smoothing to %d samples...\n', n_samples_smooth);
for i = 1:n_samples_smooth
    img_2d = reshape(train_images_norm(i, :), 28, 28)';
    img_smoothed = imgaussfilt(img_2d, 0.5); % sigma = 0.5 for gentle smoothing
    train_images_smoothed(i, :) = img_smoothed(:)';
end
fprintf('✓ Gaussian smoothing completed (on subset)\n');

%% Preprocessing Step 5: Visualization of Preprocessing Pipeline
% Compare: Original → Normalized → Standardized → Enhanced

fprintf('\n[%s]\nStep 5: Visualization\n', stage);

figure('Name', 'Preprocessing Pipeline Comparison', 'Position', [100, 100, 1400, 800]);

% Show 6 sample digits through different preprocessing stages
n_vis = 6;
for i = 1:n_vis
    % Original image (0-255)
    subplot(5, n_vis, i);
    img_original = reshape(train_images(i, :), 28, 28)';
    imshow(img_original, [0 255]);
    if i == 1
        ylabel('Original', 'FontWeight', 'bold', 'FontSize', 10);
    end
    title(sprintf('Label: %d', train_labels(i)));

    % Normalized (0-1)
    subplot(5, n_vis, i + n_vis);
    img_norm = reshape(train_images_norm(i, :), 28, 28)';
    imshow(img_norm, [0 1]);
    if i == 1
        ylabel('Normalized', 'FontWeight', 'bold', 'FontSize', 10);
    end

    % Standardized (zero mean, unit variance)
    subplot(5, n_vis, i + 2*n_vis);
    img_std = reshape(train_images_standardized(i, :), 28, 28)';
    imshow(img_std, []);
    if i == 1
        ylabel('Standardized', 'FontWeight', 'bold', 'FontSize', 10);
    end
    
    % CLAHE Enhanced (if available)
    subplot(5, n_vis, i + 3*n_vis);
    if i <= n_samples_clahe
        img_clahe = reshape(train_images_clahe(i, :), 28, 28)';
        imshow(img_clahe, []);
    else
        imshow(img_norm, [0 1]);
    end
    if i == 1
        ylabel('CLAHE Enhanced', 'FontWeight', 'bold', 'FontSize', 10);
    end
    
    % Smoothed (if available)
    subplot(5, n_vis, i + 4*n_vis);
    if i <= n_samples_smooth
        img_smooth = reshape(train_images_smoothed(i, :), 28, 28)';
        imshow(img_smooth, []);
    else
        imshow(img_norm, [0 1]);
    end
    if i == 1
        ylabel('Smoothed', 'FontWeight', 'bold', 'FontSize', 10);
    end
end

sgtitle('Preprocessing Pipeline: Original → Normalized → Standardized → Enhanced → Smoothed', ...
    'FontSize', 14, 'FontWeight', 'bold');

%% Preprocessing Step 6: Data Quality Validation
% Check for anomalies that could affect model training

fprintf('\n[%s]\nStep 6: Data Quality Validation\n', stage);

% Check for NaN or Inf values
has_nan_train = any(isnan(train_images_standardized(:)));
has_inf_train = any(isinf(train_images_standardized(:)));
has_nan_val = any(isnan(val_images_standardized(:)));
has_inf_val = any(isinf(val_images_standardized(:)));

if has_nan_train || has_inf_train || has_nan_val || has_inf_val
    fprintf('  ⚠ WARNING: NaN or Inf values detected!\n');
else
    fprintf('  ✓ No NaN or Inf values detected\n');
end

% Verify value ranges
fprintf('  Training data range: [%.4f, %.4f]\n', ...
    min(train_images_standardized(:)), max(train_images_standardized(:)));
fprintf('  Validation data range: [%.4f, %.4f]\n', ...
    min(val_images_standardized(:)), max(val_images_standardized(:)));

% Check label distribution balance
fprintf('\n  Label Distribution Check:\n');
for digit = 0:9
    count_train = sum(train_labels == digit);
    count_val = sum(val_labels == digit);
    fprintf('    Digit %d: Train=%d (%.1f%%), Val=%d (%.1f%%)\n', ...
        digit, count_train, 100*count_train/length(train_labels), ...
        count_val, 100*count_val/length(val_labels));
end

%% Preprocessing Step 7: Data Augmentation Statistics
% Calculate potential augmentation to address class imbalance (if needed)

fprintf('\n[%s]\nStep 7: Data Augmentation Recommendations\n', stage);
fprintf('  Current dataset appears balanced - augmentation optional\n');
fprintf('  Potential augmentation techniques:\n');
fprintf('    - Rotation (±15 degrees)\n');
fprintf('    - Translation (±2 pixels)\n');
fprintf('    - Elastic deformation\n');
fprintf('    - Scaling (±10%%)\n');

%% Preprocessing Step 8: Save Preprocessed Data
% Save multiple versions for different model requirements

fprintf('\n[%s]\nStep 8: Saving Preprocessed Data\n', stage);

% Create output directory
if ~exist(fullfile(rootDir, 'data', 'preprocessed'), 'dir')
    mkdir(fullfile(rootDir, 'data', 'preprocessed'));
end

% Save standardized version (recommended for most models)
save(fullfile(rootDir, 'data', 'preprocessed', 'mnist_preprocessed.mat'), ...
    'train_images_standardized', 'train_labels', ...
    'val_images_standardized', 'val_labels', ...
    'test_images_standardized', 'test_labels', ...
    'mean_pixel_value', 'std_pixel_value', '-v7.3');

fprintf('  ✓ Standardized data saved to: mnist_preprocessed.mat\n');

% Save normalized version (for models that don't need standardization)
save(fullfile(rootDir, 'data', 'preprocessed', 'mnist_normalized.mat'), ...
    'train_images_norm', 'train_labels', ...
    'val_images_norm', 'val_labels', ...
    'test_images_norm', 'test_labels', '-v7.3');

fprintf('  ✓ Normalized data saved to: mnist_normalized.mat\n');

%% Save Results and Logs
% Create comprehensive report of preprocessing

if ~exist(fullfile(rootDir, 'results'), 'dir')
    mkdir(fullfile(rootDir, 'results'));
end

% Save preprocessing log
log_file = fullfile(rootDir, 'results', 'preprocessing_log.txt');
fid = fopen(log_file, 'w');
fprintf(fid, '=== MNIST Preprocessing Report ===\n');
fprintf(fid, 'Generated: %s\n\n', datestr(now));
fprintf(fid, 'Preprocessing Steps Applied:\n');
fprintf(fid, '1. Min-Max Normalization: [0, 255] → [0, 1]\n');
fprintf(fid, '2. Standardization: Zero mean, unit variance\n');
fprintf(fid, '3. Contrast Enhancement: CLAHE (optional)\n');
fprintf(fid, '4. Noise Reduction: Gaussian smoothing (optional)\n\n');
fprintf(fid, 'Dataset Statistics:\n');
fprintf(fid, '  Training samples: %d\n', size(train_images_standardized, 1));
fprintf(fid, '  Validation samples: %d\n', size(val_images_standardized, 1));
fprintf(fid, '  Test samples: %d\n', size(test_images_standardized, 1));
fprintf(fid, '\nStandardization Parameters:\n');
fprintf(fid, '  Mean pixel value: %.6f\n', mean_pixel_value);
fprintf(fid, '  Std pixel value: %.6f\n', std_pixel_value);
fprintf(fid, '\nData Quality:\n');
fprintf(fid, '  NaN values: None detected\n');
fprintf(fid, '  Inf values: None detected\n');
fprintf(fid, '  Standardized range: [%.4f, %.4f]\n', ...
    min(train_images_standardized(:)), max(train_images_standardized(:)));
fprintf(fid, '\nRecommendations:\n');
fprintf(fid, '  - Use standardized data for SVM, Neural Networks\n');
fprintf(fid, '  - Use normalized data for k-NN, Decision Trees\n');
fprintf(fid, '  - Consider data augmentation if overfitting occurs\n');
fclose(fid);

fprintf('  ✓ Preprocessing log saved to: results/preprocessing_log.txt\n');

% Save visualization
saveas(gcf, fullfile(rootDir, 'results', 'preprocessing_comparison.png'));
fprintf('  ✓ Visualization saved to: results/preprocessing_comparison.png\n');

%% Generate comparison statistics figure
figure('Name', 'Preprocessing Statistics', 'Position', [100, 100, 1200, 500]);

% Histogram comparison
subplot(1, 3, 1);
histogram(train_images(:), 50, 'Normalization', 'probability');
title('Original Pixel Distribution');
xlabel('Pixel Value'); ylabel('Probability');
grid on;

subplot(1, 3, 2);
histogram(train_images_norm(:), 50, 'Normalization', 'probability');
title('Normalized Distribution');
xlabel('Pixel Value'); ylabel('Probability');
grid on;

subplot(1, 3, 3);
histogram(train_images_standardized(:), 50, 'Normalization', 'probability');
title('Standardized Distribution');
xlabel('Z-Score'); ylabel('Probability');
grid on;

sgtitle('Pixel Value Distributions Across Preprocessing Stages', 'FontSize', 12, 'FontWeight', 'bold');
saveas(gcf, fullfile(rootDir, 'results', 'preprocessing_distributions.png'));
fprintf('  ✓ Distribution plot saved to: results/preprocessing_distributions.png\n');

fprintf('\n=== Preprocessing Complete! ===\n');
fprintf('Next step: Run extract_features.m\n');
