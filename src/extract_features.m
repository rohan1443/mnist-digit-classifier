%% extract_features.m
% Advanced feature extraction for handwritten digit recognition
% Implements multiple feature extraction techniques:
% 1. Raw Pixels (Baseline)
% 2. Principal Component Analysis (PCA) - Dimensionality Reduction
% 3. Histogram of Oriented Gradients (HOG) - Edge/Shape features
% 4. Local Binary Patterns (LBP) - Texture features
% 5. Statistical Features - Moments, density, symmetry

clc;
fprintf('=== Feature Extraction Pipeline ===\n');

stage = 'STAGE-FEATURE-EXTRACTION';

%% Load preprocessed data
fprintf('[%s] Loading preprocessed data...\n', stage);

% Get the root directory (parent of src folder if we're in src)
currentDir = pwd;
if endsWith(currentDir, 'src')
    rootDir = fileparts(currentDir);  % Go up one level to project root
else
    rootDir = currentDir;  % Already in project root
end

% Load both normalized and standardized versions
load(fullfile(rootDir, 'data', 'preprocessed', 'mnist_normalized.mat'));
load(fullfile(rootDir, 'data', 'preprocessed', 'mnist_preprocessed.mat'));

fprintf('[%s] Data loaded successfully\n', stage);
fprintf('  Training: %d samples\n', size(train_images_norm, 1));
fprintf('  Validation: %d samples\n', size(val_images_norm, 1));
fprintf('  Test: %d samples\n', size(test_images_norm, 1));

%% Feature Set 1: Raw Pixels (Baseline)
% Direct pixel values - simplest feature representation
% Useful for: k-NN, Neural Networks

fprintf('\n[%s]\nFeature Set 1: Raw Pixel Features\n', stage);

% Use normalized data for raw pixels
train_features_raw = train_images_norm;
val_features_raw = val_images_norm;
test_features_raw = test_images_norm;

fprintf('  ✓ Raw pixel features ready\n');
fprintf('    Dimensions: %d features per image\n', size(train_features_raw, 2));

%% Feature Set 2: PCA Features (Dimensionality Reduction)
% Reduces 784 dimensions to ~50-100 while preserving 95%+ variance
% Benefits: Faster training, reduced overfitting, noise reduction

fprintf('\n[%s]\nFeature Set 2: PCA Features\n', stage);

% Apply PCA on training data
fprintf('  Computing PCA...\n');
variance_threshold = 0.95; % Preserve 95% of variance

[coeff, score, ~, ~, explained, mu] = pca(train_images_standardized);

% Find number of components needed for variance threshold
cumulative_variance = cumsum(explained) / 100;
n_components = find(cumulative_variance >= variance_threshold, 1);

fprintf('  ✓ PCA computed\n');
fprintf('    Components for %.1f%% variance: %d (from 784)\n', ...
    variance_threshold*100, n_components);
fprintf('    Dimensionality reduction: %.1f%%\n', (1 - n_components/784)*100);

% Project all datasets onto principal components
train_features_pca = (train_images_standardized - mu) * coeff(:, 1:n_components);
val_features_pca = (val_images_standardized - mu) * coeff(:, 1:n_components);
test_features_pca = (test_images_standardized - mu) * coeff(:, 1:n_components);

% Save PCA parameters for later use
pca_params.coeff = coeff(:, 1:n_components);
pca_params.mu = mu;
pca_params.n_components = n_components;
pca_params.explained_variance = explained(1:n_components);

%% Feature Set 3: HOG Features (Histogram of Oriented Gradients)
% Captures edge orientations and local shape information
% Excellent for: SVM, Random Forests

fprintf('\n[%s]\nFeature Set 3: HOG Features\n', stage);

% HOG parameters optimized for 28x28 digit images
hog_cell_size = [4 4];  % 4x4 pixel cells
hog_block_size = [2 2]; % 2x2 cell blocks
hog_num_bins = 9;        % 9 orientation bins

fprintf('  Computing HOG features...\n');
fprintf('    Cell size: %dx%d, Bins: %d\n', hog_cell_size(1), hog_cell_size(2), hog_num_bins);

% Extract HOG features for all datasets
train_features_hog = extract_hog_features(train_images_norm, hog_cell_size, hog_num_bins);
val_features_hog = extract_hog_features(val_images_norm, hog_cell_size, hog_num_bins);
test_features_hog = extract_hog_features(test_images_norm, hog_cell_size, hog_num_bins);

fprintf('  ✓ HOG features extracted\n');
fprintf('    Dimensions: %d features per image\n', size(train_features_hog, 2));

%% Feature Set 4: LBP Features (Local Binary Patterns)
% Texture-based features, robust to illumination changes
% Good for: SVM, k-NN

fprintf('\n[%s]\nFeature Set 4: LBP Features\n', stage);

fprintf('  Computing LBP features...\n');

% Extract LBP features for all datasets
train_features_lbp = extract_lbp_features(train_images_norm);
val_features_lbp = extract_lbp_features(val_images_norm);
test_features_lbp = extract_lbp_features(test_images_norm);

fprintf('  ✓ LBP features extracted\n');
fprintf('    Dimensions: %d features per image\n', size(train_features_lbp, 2));

%% Feature Set 5: Statistical Features
% Custom engineered features: moments, symmetry, density

fprintf('\n[%s]\nFeature Set 5: Statistical Features\n', stage);

fprintf('  Computing statistical features...\n');

% Extract statistical features
train_features_stats = extract_statistical_features(train_images_norm);
val_features_stats = extract_statistical_features(val_images_norm);
test_features_stats = extract_statistical_features(test_images_norm);

fprintf('  ✓ Statistical features extracted\n');
fprintf('    Dimensions: %d features per image\n', size(train_features_stats, 2));

%% Feature Set 6: Combined Features
% Concatenate multiple feature types for ensemble learning

fprintf('\n[%s]\nFeature Set 6: Combined Features\n', stage);

% Combine HOG + LBP + Statistical features
train_features_combined = [train_features_hog, train_features_lbp, train_features_stats];
val_features_combined = [val_features_hog, val_features_lbp, val_features_stats];
test_features_combined = [test_features_hog, test_features_lbp, test_features_stats];

fprintf('  ✓ Combined features created\n');
fprintf('    Total dimensions: %d features\n', size(train_features_combined, 2));

%% Visualization: Feature Space Analysis

fprintf('\n[%s]\nGenerating visualizations...\n', stage);

% 1. PCA Variance Explained
figure('Name', 'PCA Analysis', 'Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
plot(cumulative_variance(1:min(100, length(cumulative_variance))), 'LineWidth', 2);
hold on;
yline(variance_threshold, 'r--', 'LineWidth', 2);
plot(n_components, variance_threshold, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Number of Components');
ylabel('Cumulative Explained Variance');
title('PCA: Variance Explained');
grid on;
legend('Cumulative Variance', sprintf('%.0f%% Threshold', variance_threshold*100), ...
    sprintf('%d Components', n_components), 'Location', 'southeast');

% 2. First 3 Principal Components
subplot(1, 3, 2);
% Sample 1000 points for visualization
n_vis = min(1000, size(train_features_pca, 1));
idx_vis = randperm(size(train_features_pca, 1), n_vis);
scatter3(train_features_pca(idx_vis, 1), train_features_pca(idx_vis, 2), ...
    train_features_pca(idx_vis, 3), 10, train_labels(idx_vis), 'filled');
xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
title('First 3 Principal Components');
colorbar; colormap(jet);
grid on; view(45, 30);

% 3. Top Principal Components Visualization
subplot(1, 3, 3);
% Visualize first 6 principal components as images
for i = 1:6
    subplot(2, 6, i+6);
    pc_img = reshape(coeff(:, i), 28, 28)';
    imagesc(pc_img); axis image; axis off;
    title(sprintf('PC%d', i), 'FontSize', 8);
    colormap(gray);
end

sgtitle('PCA Feature Space Analysis', 'FontSize', 14, 'FontWeight', 'bold');

% 2. Feature Distribution Comparison
figure('Name', 'Feature Distributions', 'Position', [100, 100, 1400, 800]);

feature_types = {'Raw Pixels', 'PCA', 'HOG', 'LBP', 'Statistical', 'Combined'};
feature_sets = {train_features_raw(:), train_features_pca(:), train_features_hog(:), ...
    train_features_lbp(:), train_features_stats(:), train_features_combined(:)};

for i = 1:6
    subplot(2, 3, i);
    histogram(feature_sets{i}, 50, 'Normalization', 'probability');
    title(feature_types{i});
    xlabel('Feature Value'); ylabel('Probability');
    grid on;
end

sgtitle('Feature Value Distributions', 'FontSize', 14, 'FontWeight', 'bold');

%% Save Extracted Features

fprintf('\n[%s]\nSaving extracted features...\n', stage);

% Create features directory
if ~exist(fullfile(rootDir, 'data', 'features'), 'dir')
    mkdir(fullfile(rootDir, 'data', 'features'));
end

% Save all feature sets
save(fullfile(rootDir, 'data', 'features', 'features_raw.mat'), ...
    'train_features_raw', 'val_features_raw', 'test_features_raw', ...
    'train_labels', 'val_labels', 'test_labels', '-v7.3');

save(fullfile(rootDir, 'data', 'features', 'features_pca.mat'), ...
    'train_features_pca', 'val_features_pca', 'test_features_pca', ...
    'train_labels', 'val_labels', 'test_labels', 'pca_params', '-v7.3');

save(fullfile(rootDir, 'data', 'features', 'features_hog.mat'), ...
    'train_features_hog', 'val_features_hog', 'test_features_hog', ...
    'train_labels', 'val_labels', 'test_labels', '-v7.3');

save(fullfile(rootDir, 'data', 'features', 'features_lbp.mat'), ...
    'train_features_lbp', 'val_features_lbp', 'test_features_lbp', ...
    'train_labels', 'val_labels', 'test_labels', '-v7.3');

save(fullfile(rootDir, 'data', 'features', 'features_statistical.mat'), ...
    'train_features_stats', 'val_features_stats', 'test_features_stats', ...
    'train_labels', 'val_labels', 'test_labels', '-v7.3');

save(fullfile(rootDir, 'data', 'features', 'features_combined.mat'), ...
    'train_features_combined', 'val_features_combined', 'test_features_combined', ...
    'train_labels', 'val_labels', 'test_labels', '-v7.3');

fprintf('  ✓ All feature sets saved to: data/features/\n');

%% Save Results and Logs

if ~exist(fullfile(rootDir, 'results'), 'dir')
    mkdir(fullfile(rootDir, 'results'));
end

% Save feature extraction log
log_file = fullfile(rootDir, 'results', 'feature_extraction_log.txt');
fid = fopen(log_file, 'w');
fprintf(fid, '=== Feature Extraction Report ===\n');
fprintf(fid, 'Generated: %s\n\n', datestr(now));
fprintf(fid, 'Feature Sets Extracted:\n\n');
fprintf(fid, '1. Raw Pixels\n');
fprintf(fid, '   Dimensions: %d\n', size(train_features_raw, 2));
fprintf(fid, '   Description: Direct normalized pixel values\n');
fprintf(fid, '   Best for: k-NN, Neural Networks\n\n');
fprintf(fid, '2. PCA Features\n');
fprintf(fid, '   Dimensions: %d (reduced from 784)\n', n_components);
fprintf(fid, '   Variance preserved: %.2f%%\n', variance_threshold*100);
fprintf(fid, '   Description: Principal component analysis\n');
fprintf(fid, '   Best for: SVM, Neural Networks (faster training)\n\n');
fprintf(fid, '3. HOG Features\n');
fprintf(fid, '   Dimensions: %d\n', size(train_features_hog, 2));
fprintf(fid, '   Parameters: Cell=%dx%d, Bins=%d\n', hog_cell_size(1), hog_cell_size(2), hog_num_bins);
fprintf(fid, '   Description: Histogram of Oriented Gradients\n');
fprintf(fid, '   Best for: SVM, Random Forest\n\n');
fprintf(fid, '4. LBP Features\n');
fprintf(fid, '   Dimensions: %d\n', size(train_features_lbp, 2));
fprintf(fid, '   Description: Local Binary Patterns (texture)\n');
fprintf(fid, '   Best for: SVM, k-NN\n\n');
fprintf(fid, '5. Statistical Features\n');
fprintf(fid, '   Dimensions: %d\n', size(train_features_stats, 2));
fprintf(fid, '   Description: Moments, symmetry, density\n');
fprintf(fid, '   Best for: Ensemble methods\n\n');
fprintf(fid, '6. Combined Features\n');
fprintf(fid, '   Dimensions: %d\n', size(train_features_combined, 2));
fprintf(fid, '   Description: HOG + LBP + Statistical\n');
fprintf(fid, '   Best for: SVM, Ensemble methods\n\n');
fprintf(fid, 'Recommendations:\n');
fprintf(fid, '  - Use PCA for fast prototyping\n');
fprintf(fid, '  - Use HOG for SVM classifier\n');
fprintf(fid, '  - Use Combined features for best accuracy\n');
fclose(fid);

fprintf('  ✓ Feature extraction log saved\n');

% Save visualizations
saveas(figure(1), fullfile(rootDir, 'results', 'pca_analysis.png'));
saveas(figure(2), fullfile(rootDir, 'results', 'feature_distributions.png'));
fprintf('  ✓ Visualizations saved\n');

fprintf('\n=== Feature Extraction Complete! ===\n');
fprintf('Next step: Run train_model.m\n');

%% Helper Functions

function hog_features = extract_hog_features(images, cell_size, num_bins)
    % Extract HOG features for all images
    n_samples = size(images, 1);
    
    % Get HOG feature size from first image
    test_img = reshape(images(1, :), 28, 28)';
    test_hog = extractHOGFeatures(test_img, 'CellSize', cell_size, 'NumBins', num_bins);
    hog_dim = length(test_hog);
    
    % Preallocate
    hog_features = zeros(n_samples, hog_dim);
    
    % Extract features with progress indicator
    for i = 1:n_samples
        img = reshape(images(i, :), 28, 28)';
        hog_features(i, :) = extractHOGFeatures(img, 'CellSize', cell_size, 'NumBins', num_bins);
        
        if mod(i, 5000) == 0
            fprintf('    Processed %d/%d images...\n', i, n_samples);
        end
    end
end

function lbp_features = extract_lbp_features(images)
    % Extract LBP features for all images
    n_samples = size(images, 1);
    
    % Get LBP feature size from first image
    test_img = reshape(images(1, :), 28, 28)';
    test_lbp = extractLBPFeatures(test_img);
    lbp_dim = length(test_lbp);
    
    % Preallocate
    lbp_features = zeros(n_samples, lbp_dim);
    
    % Extract features with progress indicator
    for i = 1:n_samples
        img = reshape(images(i, :), 28, 28)';
        lbp_features(i, :) = extractLBPFeatures(img);
        
        if mod(i, 5000) == 0
            fprintf('    Processed %d/%d images...\n', i, n_samples);
        end
    end
end

function stat_features = extract_statistical_features(images)
    % Extract custom statistical features
    n_samples = size(images, 1);
    
    % 14 statistical features per image
    stat_features = zeros(n_samples, 14);
    
    for i = 1:n_samples
        img = reshape(images(i, :), 28, 28)';
        
        % Moment features
        stat_features(i, 1) = mean(img(:));           % Mean intensity
        stat_features(i, 2) = std(img(:));            % Standard deviation
        stat_features(i, 3) = skewness(img(:));       % Skewness
        stat_features(i, 4) = kurtosis(img(:));       % Kurtosis
        
        % Density features
        threshold = 0.5;
        binary_img = img > threshold;
        stat_features(i, 5) = sum(binary_img(:)) / numel(img);  % Pixel density
        
        % Symmetry features (horizontal and vertical)
        img_flip_h = fliplr(img);
        img_flip_v = flipud(img);
        stat_features(i, 6) = corr2(img, img_flip_h);  % Horizontal symmetry
        stat_features(i, 7) = corr2(img, img_flip_v);  % Vertical symmetry
        
        % Projection features
        h_proj = sum(img, 1);  % Horizontal projection
        v_proj = sum(img, 2);  % Vertical projection
        stat_features(i, 8) = std(h_proj);   % Horizontal projection std
        stat_features(i, 9) = std(v_proj);   % Vertical projection std
        
        % Gradient features
        [Gx, Gy] = gradient(img);
        stat_features(i, 10) = mean(abs(Gx(:)));   % Mean horizontal gradient
        stat_features(i, 11) = mean(abs(Gy(:)));   % Mean vertical gradient
        
        % Entropy (information content)
        stat_features(i, 12) = entropy(img);
        
        % Edge density
        edges = edge(img, 'Canny');
        stat_features(i, 13) = sum(edges(:)) / numel(edges);
        
        % Aspect ratio of bounding box
        [rows, cols] = find(binary_img);
        if ~isempty(rows)
            height = max(rows) - min(rows) + 1;
            width = max(cols) - min(cols) + 1;
            stat_features(i, 14) = height / width;
        else
            stat_features(i, 14) = 1;
        end
        
        if mod(i, 5000) == 0
            fprintf('    Processed %d/%d images...\n', i, n_samples);
        end
    end
end
