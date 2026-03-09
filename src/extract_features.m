%% extract_features.m
% Feature extraction pipeline (NO timestamps)
% Input:  data/preprocessed/mnist_preprocessed.mat
% Output: Simple feature files without timestamps

stage = 'STAGE-FEATURE-EXTRACTION';
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  %s                              ║\n', stage);
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

rootDir = fileparts(fileparts(mfilename('fullpath')));

% === STEP 1: Load preprocessed data ===
preprocFile = fullfile(rootDir, 'data', 'preprocessed', 'mnist_preprocessed.mat');
if ~exist(preprocFile, 'file')
    error('❌ Preprocessed data not found!\nRun src/preprocess_data.m first.');
end

fprintf('[%s] Loading preprocessed data...\n', stage);
load(preprocFile);
fprintf('[%s] ✓ Loaded:\n', stage);
fprintf('     Training:   %d samples\n', size(train_images_centered, 1));
fprintf('     Validation: %d samples\n', size(val_images_centered, 1));
fprintf('     Test:       %d samples\n', size(test_images_centered, 1));

% === STEP 2: Create feature directories ===
featDirs = {'raw', 'hog', 'pca'};
for i = 1:length(featDirs)
    dirPath = fullfile(rootDir, 'data', 'features', featDirs{i});
    if ~exist(dirPath, 'dir')
        mkdir(dirPath);
        fprintf('[%s] Created directory: %s\n', stage, dirPath);
    end
end

% === METHOD 1: RAW PIXELS (baseline) ===
fprintf('\n[%s] Extracting RAW PIXELS (baseline)\n', stage);
X_train_raw = train_images_centered;
X_val_raw   = val_images_centered;
X_test_raw  = test_images_centered;

save(fullfile(rootDir, 'data', 'features', 'raw', 'features_raw.mat'), ...
    'X_train_raw', 'X_val_raw', 'X_test_raw', ...
    'train_labels', 'val_labels', 'test_labels');
fprintf('[%s] ✓ Raw pixels saved (%d dimensions)\n', stage, size(X_train_raw, 2));

% === METHOD 2: HOG FEATURES ===
fprintf('\n[%s] Extracting HOG features (Cell=7x7, numBins=9)...\n', stage);

% Reshape centered vectors back to 28x28 images for HOG
% NOTE: Must ADD BACK mean_pixel_value to get valid grayscale images for HOG
train_images_28 = zeros(28, 28, size(train_images_centered, 1));
val_images_28   = zeros(28, 28, size(val_images_centered, 1));
test_images_28  = zeros(28, 28, size(test_images_centered, 1));

for i = 1:size(train_images_centered, 1)
    img_norm = reshape(train_images_centered(i,:), 28, 28)' + mean_pixel_value;
    img_255 = uint8(img_norm * 255);
    train_images_28(:, :, i) = img_255;
end
for i = 1:size(val_images_centered, 1)
    img_norm = reshape(val_images_centered(i,:), 28, 28)' + mean_pixel_value;
    img_255 = uint8(img_norm * 255);
    val_images_28(:, :, i) = img_255;
end
for i = 1:size(test_images_centered, 1)
    img_norm = reshape(test_images_centered(i,:), 28, 28)' + mean_pixel_value;
    img_255 = uint8(img_norm * 255);
    test_images_28(:, :, i) = img_255;
end

% Extract HOG features
fprintf('  Extracting HOG from training set...\n');
X_train_hog = zeros(size(train_images_centered, 1), 0);
for i = 1:size(train_images_centered, 1)
    if mod(i, 5000) == 0
        fprintf('    Processed %d/%d...\n', i, size(train_images_centered, 1));
    end
    hog_feat = extractHOGFeatures(train_images_28(:, :, i), ...
        'CellSize', [7 7], 'BlockSize', [2 2], 'numBins', 9);
    if i == 1
        X_train_hog = zeros(size(train_images_centered, 1), length(hog_feat));
    end
    X_train_hog(i, :) = hog_feat;
end

fprintf('  Extracting HOG from validation set...\n');
X_val_hog = zeros(size(val_images_centered, 1), size(X_train_hog, 2));
for i = 1:size(val_images_centered, 1)
    hog_feat = extractHOGFeatures(val_images_28(:, :, i), ...
        'CellSize', [7 7], 'BlockSize', [2 2], 'numBins', 9);
    X_val_hog(i, :) = hog_feat;
end

fprintf('  Extracting HOG from test set...\n');
X_test_hog = zeros(size(test_images_centered, 1), size(X_train_hog, 2));
for i = 1:size(test_images_centered, 1)
    hog_feat = extractHOGFeatures(test_images_28(:, :, i), ...
        'CellSize', [7 7], 'BlockSize', [2 2], 'numBins', 9);
    X_test_hog(i, :) = hog_feat;
end

% Mean center HOG features (using training mean)
mean_hog = mean(X_train_hog, 1);
X_train_hog = X_train_hog - mean_hog;
X_val_hog   = X_val_hog - mean_hog;
X_test_hog  = X_test_hog - mean_hog;

save(fullfile(rootDir, 'data', 'features', 'hog', 'features_hog.mat'), ...
    'X_train_hog', 'X_val_hog', 'X_test_hog', ...
    'train_labels', 'val_labels', 'test_labels');
fprintf('[%s] ✓ HOG features saved (%d dimensions)\n', stage, size(X_train_hog, 2));

% === METHOD 3: PCA FEATURES (95% variance) ===
fprintf('\n[%s] Extracting PCA features (95%% variance)...\n', stage);

% Suppress harmless linear dependency warning
warning('off', 'stats:pca:tsqNumComponents');
[coeff, score, ~, ~, explained] = pca(train_images_centered);
warning('on', 'stats:pca:tsqNumComponents');

cumvar = cumsum(explained);
n_pca = find(cumvar >= 95, 1, 'first');

X_train_pca = score(:, 1:n_pca);
% FIX: Use centered data directly, do NOT subtract mean_pixel_value again
X_val_pca   = val_images_centered * coeff(:, 1:n_pca);
X_test_pca  = test_images_centered * coeff(:, 1:n_pca);

save(fullfile(rootDir, 'data', 'features', 'pca', 'features_pca.mat'), ...
    'X_train_pca', 'X_val_pca', 'X_test_pca', ...
    'train_labels', 'val_labels', 'test_labels', ...
    'n_pca', 'explained');
fprintf('[%s] ✓ PCA features saved (%d dimensions, %.1f%% variance)\n', stage, n_pca, cumvar(n_pca));

% === SUMMARY ===
fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  FEATURE EXTRACTION COMPLETE                              ║\n');
fprintf('╠════════════════════════════════════════════════════════════╣\n');
fprintf('║  Method │ Dimensions │ Output File                      ║\n');
fprintf('╠════════════════════════════════════════════════════════════╣\n');
fprintf('║  Raw    │ %4d       │ data/features/raw/features_raw.mat ║\n', size(X_train_raw,2));
fprintf('║  HOG    │ %4d       │ data/features/hog/features_hog.mat ║\n', size(X_train_hog,2));
fprintf('║  PCA    │ %4d       │ data/features/pca/features_pca.mat ║\n', n_pca);
fprintf('╚════════════════════════════════════════════════════════════╝\n');
fprintf('\nNEXT STEP: Run classifier training scripts\n');