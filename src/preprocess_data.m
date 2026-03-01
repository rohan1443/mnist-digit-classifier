clc;

fprintf("=== Preprocessing Data ===\n");

stage = 'STAGE-PREPROCESSING';

fprintf('[%s] Loading the raw data from the .mat file...\n', stage);
rootDir = pwd;
fprintf('Root directory: %s\n', rootDir);
load(fullfile(rootDir, 'data', 'loaded', 'mnist_data.mat'));
% load: loads all the variables from .mat file into this workspace

fprintf('[%s] Data loading done\n', stage);

% to understand the %s and %d are placeholders for matlab to know where to print, else gives jibbrish output
fprintf('[%s] train Images size: %d x %d\n', stage, size(train_images,1 ), size(train_images,2));
fprintf('[%s] validation images size: %d x %d\n', stage, size(val_images,1 ), size(val_images,2));
fprintf('[%s] test images size: %d x %d\n', stage, size(test_images,1 ), size(test_images,2));


%% PreProcessing step 1: Normalization
% Normalization of pixel values to [0, 1] range as most ML models perform better with normalized data
% so i will be dividing all the pixel values by 255 (the maximum pixel value in grayscale images) to scale them to the [0, 1] range

fprintf('\n\n[%s]\n [%s]', stage, 'Step 1: Normalizing');
train_images_norm = double(train_images) / 255;
val_images_norm = double(val_images) / 255;
test_images_norm = double(test_images) / 255;

fprintf('Normalization done\n');
fprintf('New Pixel Range: [%.2f, %.2f] \n', min(train_images_norm(:)), max(train_images_norm(:)));


%% PreProcessing step 2: Mean Centering
% Mean centering the data as it can help some models converge faster during training by centering the data around zero
% i will be calculating the mean pixel value across the training set and then subtracting this mean

fprintf('\n\n[%s]\n [%s]', stage, 'Step 2: Mean Centering for better convergence training data');

% Calculate mean pixel value across the training set
mean_pixel_value = mean(train_images_norm(:)); % mean: calculates the average value of all pixels in the training set

% Now subtracting the mean from all the dataset variables to center the data around zero
train_images_centered = train_images_norm - mean_pixel_value;
val_images_centered = val_images_norm - mean_pixel_value;
test_images_centered = test_images_norm - mean_pixel_value;

fprintf('Mean centering done\n');
fprintf('Mean pixel value (training set): %.4f\n', mean_pixel_value);
fprintf('New Pixel Range after mean centering: [%.2f, %.2f] \n', min(train_images_centered(:)), max(train_images_centered(:)));

%% PreProcessing step 3: Visualization of preprocessed data
% Visualize some preprocessed images to verify the transformations by comparing the original and preprocessed images side by side

figure('Name', 'Preprocessed MNIST Digits'); % figure: creates new window for visualization

% Show 6 samples
for i = 1:6 % Loop to display 6 images (i from 1,2,3,4,5,6)
    % Original image (0-255)
    subplot(3, 6, i); % subplot: creates a 3x6 grid layout (3 rows, 6 columns) and places the current plot in the i-th position
    img_original = reshape(train_images(i, :), 28, 28)'; % reshape: converts 1D array back to 2D image (28x28), transpose (') fixes orientation because matlab fills images column-wise and here images are stored row-wise
    imshow(img_original, [0 255]); % Display with pixel intensity range 0-255
    if i == 1 % Add ylabel only to the first column of images for clarity in the row
        ylabel('Original', 'FontWeight', 'bold');
    end
    title(sprintf('Label: %d', train_labels(i)));

    % Normalized image (0-1)
    subplot(3, 6, i+6);  % Place the normalized images in the second row (i+6 means positions 7-12)
    img_norm = reshape(train_images_norm(i, :), 28, 28)';
    imshow(img_norm, [0 1]); % Display with 0-1 range
    if i == 1 % similarly add ylabel only to the first column of images for clarity in the 2nd row
        ylabel('Normalized', 'FontWeight', 'bold');
    end

    % Centered image (zero mean)
    subplot(3, 6, i+12); % Place the mean centered images in the third row (i+12 means positions 13-18)
    img_centered = reshape(train_images_centered(i, :), 28, 28)';
    imshow(img_centered, []); % Auto-scale for centered data (can be negative)
    if i == 1 % similarly add ylabel only to the first column of images for clarity in the 3rd row
        ylabel('Centered', 'FontWeight', 'bold');
    end
end

sgtitle('Preprocessing Steps Comparison'); % sgtitle: super title for entire figure


%% PreProcessing Step 4: Data Quality Check after normalization and mean centering
% Check for any NaN or Inf values in the preprocessed datasets which can cause issues during model training
fprintf('\n\n[%s]\n [%s]', stage, 'Data Quality Check after Preprocessing');

% Check for NaN values

if any(isnan(train_images_norm(:))) || any(isinf(train_images_norm(:)))
    fprintf('Warning: NaN or Inf values found in normalized training images data!\n');
else
    fprintf('No NaN or Inf values in normalized training images data.\n');
end

%% PreProcessing Step 5: Save the preprocessed data for later use in model training and evaluation

fprintf('\n Saving the preprocessed data to a new .mat file for use in the next stage \n');

save(fullfile(rootDir, 'data', 'preprocessed', 'mnist_preprocessed.mat'), ... % the file locaiton is the preprocessed folder inside the data folder
    'train_images_centered', 'train_labels', ...
    'val_images_centered', 'val_labels', ...
    'test_images_centered', 'test_labels', '-v7')

fprintf('Preprocessed data saved successfully to mnist_preprocessed.mat\n');

