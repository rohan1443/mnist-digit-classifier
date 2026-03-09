clc;

fprintf("=== Preprocessing Data ===\n");

stage = 'STAGE-PREPROCESSING';

fprintf('[%s] Loading the raw data...\n', stage);
rootDir = fileparts(fileparts(mfilename('fullpath')));
fprintf('Root directory: %s\n', rootDir);

% Try to load from the pre-saved .mat file (created by load_data.m)
matFile = fullfile(rootDir, 'data', 'loaded', 'mnist_data.mat');
if exist(matFile, 'file')
    fprintf('[%s] Found saved data file: %s\n', stage, matFile);
    load(matFile);
else
    fprintf('[%s] Saved data file not found. Loading from CSV instead...\n', stage);
    % Fallback: load CSV directly (this duplicates load_data.m functionality)
    train_data = readmatrix(fullfile(rootDir, 'data', 'csv', 'mnist_train.csv'));
    test_data  = readmatrix(fullfile(rootDir, 'data', 'csv', 'mnist_test.csv'));

    train_labels = train_data(:, 1);
    train_images = train_data(:, 2:end);
    test_labels  = test_data(:, 1);
    test_images  = test_data(:, 2:end);
end

% Verify that the required variables exist
if ~exist('train_images', 'var') || ~exist('train_labels', 'var') || ...
   ~exist('test_images', 'var')  || ~exist('test_labels', 'var')
    error('Required data variables (train_images, train_labels, test_images, test_labels) not found.');
end

fprintf('[%s] Data loading done\n', stage);

% === Create validation split if it doesn't exist ===
if ~exist('val_images', 'var')
    fprintf('[%s] Creating validation split from training data...\n', stage);
    val_images = train_images(48001:end, :);
    val_labels = train_labels(48001:end, :);
    train_images = train_images(1:48000, :);
    train_labels = train_labels(1:48000, :);
    fprintf('[%s] Validation split created: %d samples\n', stage, size(val_images, 1));
end

fprintf('[%s] train Images size: %d x %d\n', stage, size(train_images,1), size(train_images,2));
fprintf('[%s] validation images size: %d x %d\n', stage, size(val_images,1), size(val_images,2));
fprintf('[%s] test images size: %d x %d\n', stage, size(test_images,1), size(test_images,2));


%% PreProcessing step 1: Normalization (min‑max scaling to [0,1])
fprintf('\n\n[%s]\n [%s]', stage, 'Step 1: Normalizing');
train_images_norm = double(train_images) / 255;
val_images_norm = double(val_images) / 255;
test_images_norm = double(test_images) / 255;

fprintf('Normalization done\n');
fprintf('New Pixel Range: [%.2f, %.2f] \n', min(train_images_norm(:)), max(train_images_norm(:)));


%% PreProcessing step 2: Global Mean Centering
fprintf('\n\n[%s]\n [%s]', stage, 'Step 2: Mean Centering');

mean_pixel_value = mean(train_images_norm(:));

train_images_centered = train_images_norm - mean_pixel_value;
val_images_centered = val_images_norm - mean_pixel_value;
test_images_centered = test_images_norm - mean_pixel_value;

fprintf('Mean centering done\n');
fprintf('Mean pixel value (training set): %.4f\n', mean_pixel_value);
fprintf('New Pixel Range after mean centering: [%.2f, %.2f] \n', min(train_images_centered(:)), max(train_images_centered(:)));


%% PreProcessing step 3: Per‑Pixel Standardization (Z‑Score)
fprintf('\n\n[%s]\n [%s]', stage, 'Step 3: Per‑Pixel Standardization (Z‑Score)');

mu_pixel = mean(train_images_norm, 1);            % 1 x 784
sigma_pixel = std(train_images_norm, 0, 1);       % 1 x 784
sigma_pixel(sigma_pixel == 0) = 1;                % Avoid division by zero

train_images_std = (train_images_norm - mu_pixel) ./ sigma_pixel;
val_images_std   = (val_images_norm   - mu_pixel) ./ sigma_pixel;
test_images_std  = (test_images_norm  - mu_pixel) ./ sigma_pixel;

fprintf('Standardization done\n');
fprintf('Each pixel now has mean ≈ 0 and std ≈ 1 on the training set.\n');
fprintf('Check first pixel: mean = %.4f, std = %.4f\n', ...
    mean(train_images_std(:,1)), std(train_images_std(:,1)));


%% PreProcessing step 4: Visualization
figure('Name', 'Preprocessed MNIST Digits');

for i = 1:6
    % Original (0-255)
    subplot(4, 6, i);
    img_original = reshape(train_images(i, :), 28, 28)';
    imshow(img_original, [0 255]);
    if i == 1, ylabel('Original', 'FontWeight', 'bold'); end
    title(sprintf('Label: %d', train_labels(i)));

    % Normalized (0-1)
    subplot(4, 6, i+6);
    img_norm = reshape(train_images_norm(i, :), 28, 28)';
    imshow(img_norm, [0 1]);
    if i == 1, ylabel('Normalized', 'FontWeight', 'bold'); end

    % Centered
    subplot(4, 6, i+12);
    img_centered = reshape(train_images_centered(i, :), 28, 28)';
    imshow(img_centered, []);
    if i == 1, ylabel('Centered', 'FontWeight', 'bold'); end

    % Standardized
    subplot(4, 6, i+18);
    img_std = reshape(train_images_std(i, :), 28, 28)';
    imshow(img_std, []);
    if i == 1, ylabel('Standardized', 'FontWeight', 'bold'); end
end
sgtitle('Preprocessing Steps Comparison');


%% PreProcessing Step 5: Data Quality Check
fprintf('\n\n[%s]\n [%s]', stage, 'Data Quality Check after Preprocessing');

if any(isnan(train_images_norm(:))) || any(isinf(train_images_norm(:)))
    fprintf('Warning: NaN or Inf values found in normalized training images data!\n');
else
    fprintf('No NaN or Inf values in normalized training images data.\n');
end

if any(isnan(train_images_std(:))) || any(isinf(train_images_std(:)))
    fprintf('Warning: NaN or Inf values found in standardized training images data!\n');
else
    fprintf('No NaN or Inf values in standardized training images data.\n');
end


%% PreProcessing Step 6: Save preprocessed data
fprintf('\n Saving the preprocessed data...\n');

if ~exist(fullfile(rootDir, 'data', 'preprocessed'), 'dir')
    mkdir(fullfile(rootDir, 'data', 'preprocessed'));
    fprintf('[%s] Created folder: data/preprocessed/\n', stage);
end

save(fullfile(rootDir, 'data', 'preprocessed', 'mnist_preprocessed.mat'), ...
    'train_images_centered', 'train_labels', ...
    'val_images_centered', 'val_labels', ...
    'test_images_centered', 'test_labels', ...
    'train_images_std', 'val_images_std', 'test_images_std', ...
    'mu_pixel', 'sigma_pixel', ...
    'mean_pixel_value', '-v7');

fprintf('Preprocessed data saved successfully to mnist_preprocessed.mat\n');
fprintf('Now includes:\n');
fprintf('  - Normalized (0-1)     : used internally\n');
fprintf('  - Mean‑centered         : *_centered\n');
fprintf('  - Per‑pixel standardized: *_std\n');
fprintf('  - Per‑pixel statistics  : mu_pixel, sigma_pixel\n');
fprintf('[%s] Preprocessing Complete!\n', stage);