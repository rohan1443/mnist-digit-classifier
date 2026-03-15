function demo_app()
% demo_app.m
% Local interactive demo for handwritten digit prediction using pre-trained models.
% No end-to-end retraining is required.

clc;
close all;

rootDir = get_project_root();
fprintf('=== MNIST Demo Application ===\n');
fprintf('Project root: %s\n', rootDir);

assets = load_demo_assets(rootDir);
ui = build_ui();

ui.statusText.Value = sprintf(['Ready.\n' ...
    '1) Click "Upload Digit Image".\n' ...
    '2) Inspect per-model prediction + confidence.']);

ui.uploadBtn.ButtonPushedFcn = @(~, ~) on_upload();
ui.sampleBtn.ButtonPushedFcn = @(~, ~) on_load_sample();

    function on_upload()
        [fileName, filePath] = uigetfile({ ...
            '*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tif;*.tiff', ...
            'Image files (*.png, *.jpg, *.jpeg, *.bmp, *.gif, *.tif, *.tiff)'}, ...
            'Select Handwritten Digit Image');

        if isequal(fileName, 0)
            return;
        end

        fullPath = fullfile(filePath, fileName);
        run_prediction_flow(fullPath);
    end

    function on_load_sample()
        sampleDir = fullfile(rootDir, 'demo', 'test_images');
        if ~exist(sampleDir, 'dir')
            uialert(ui.fig, ['Sample folder not found: ' sampleDir], ...
                'Missing Sample Images');
            return;
        end

        [fileName, filePath] = uigetfile({ ...
            '*.png;*.jpg;*.jpeg;*.bmp;*.gif', ...
            'Sample image files'}, ...
            'Select Sample Test Image', sampleDir);

        if isequal(fileName, 0)
            return;
        end

        run_prediction_flow(fullfile(filePath, fileName));
    end

    function run_prediction_flow(imagePath)
        try
            [img28, visImg] = preprocess_uploaded_image(imagePath);
            result = predict_with_all_models(img28, assets);

            imshow(visImg, 'Parent', ui.axInput);
            title(ui.axInput, 'Uploaded Image', 'FontWeight', 'bold');

            imshow(img28, 'Parent', ui.axProc);
            title(ui.axProc, 'Processed 28x28 Input', 'FontWeight', 'bold');

            tableData = {
                'SVM (HOG)', result.svm.prediction, sprintf('%.2f', result.svm.confidence * 100);
                'Random Forest (Combined)', result.rf.prediction, sprintf('%.2f', result.rf.confidence * 100);
                'k-NN (Raw Pixels)', result.knn.prediction, sprintf('%.2f', result.knn.confidence * 100)
            };
            ui.resultsTable.Data = tableData;

            cla(ui.axBars);
            confVals = [result.svm.confidence, result.rf.confidence, result.knn.confidence] * 100;
            b = bar(ui.axBars, confVals, 0.5);
            b.FaceColor = 'flat';
            b.CData = [0.20 0.45 0.80; 0.18 0.62 0.34; 0.86 0.49 0.18];
            ylim(ui.axBars, [0 100]);
            ylabel(ui.axBars, 'Confidence (%)');
            set(ui.axBars, 'XTickLabel', {'SVM', 'RF', 'k-NN'});
            grid(ui.axBars, 'on');
            title(ui.axBars, 'Per-Model Confidence', 'FontWeight', 'bold');

            [~, bestIdx] = max(confVals);
            modelNames = {'SVM', 'Random Forest', 'k-NN'};
            predictions = [result.svm.prediction, result.rf.prediction, result.knn.prediction];

            ui.statusText.Value = sprintf(['Image: %s\n' ...
                'Best confidence model: %s\n' ...
                'Predicted digit: %d\n\n' ...
                'Note: SVM confidence is softmax-normalized from ECOC scores.'], ...
                imagePath, modelNames{bestIdx}, predictions(bestIdx));

        catch ME
            ui.statusText.Value = sprintf('Failed to predict.\n%s', ME.message);
            rethrow(ME);
        end
    end
end

function assets = load_demo_assets(rootDir)
% Load trained models and ensure required files exist.

requiredFiles = {
    fullfile(rootDir, 'models', 'svm_model.mat')
    fullfile(rootDir, 'models', 'rf_model.mat')
    fullfile(rootDir, 'models', 'knn_model.mat')
};

for i = 1:numel(requiredFiles)
    if ~exist(requiredFiles{i}, 'file')
        error('Missing required model file: %s', requiredFiles{i});
    end
end

svmData = load(requiredFiles{1}, 'svm_model');
rfData = load(requiredFiles{2}, 'rf_model');
knnData = load(requiredFiles{3}, 'knn_model');

assets.svm_model = svmData.svm_model;
assets.rf_model = rfData.rf_model;
assets.knn_model = knnData.knn_model;
end

function ui = build_ui()
% Build a lightweight UI for local demos.

ui.fig = uifigure('Name', 'MNIST Pattern Recognition Demo', 'Position', [100 100 1100 650]);

ui.uploadBtn = uibutton(ui.fig, 'push', ...
    'Text', 'Upload Digit Image', ...
    'Position', [30 595 170 36], ...
    'FontWeight', 'bold');

ui.sampleBtn = uibutton(ui.fig, 'push', ...
    'Text', 'Load Sample Image', ...
    'Position', [220 595 170 36]);

ui.statusText = uitextarea(ui.fig, ...
    'Position', [420 560 650 72], ...
    'Editable', 'off');

ui.axInput = uiaxes(ui.fig, 'Position', [30 320 320 240]);
title(ui.axInput, 'Uploaded Image');
axis(ui.axInput, 'off');

ui.axProc = uiaxes(ui.fig, 'Position', [30 55 320 240]);
title(ui.axProc, 'Processed 28x28 Input');
axis(ui.axProc, 'off');

ui.resultsTable = uitable(ui.fig, ...
    'Position', [380 320 690 220], ...
    'ColumnName', {'Model', 'Predicted Digit', 'Confidence (%)'}, ...
    'ColumnEditable', [false false false], ...
    'Data', cell(3, 3));

ui.axBars = uiaxes(ui.fig, 'Position', [420 70 620 220]);
title(ui.axBars, 'Per-Model Confidence');
end

function [img28, visImg] = preprocess_uploaded_image(imagePath)
% Convert arbitrary input image to MNIST-like 28x28 white digit on black.

raw = imread(imagePath);
if size(raw, 3) == 3
    gray = rgb2gray(raw);
else
    gray = raw;
end

gray = im2double(gray);
visImg = gray;

% Heuristic: if border is bright, assume dark digit on bright paper and invert.
border = [gray(1, :), gray(end, :), gray(:, 1).', gray(:, end).'];
if mean(border) > 0.5
    gray = 1 - gray;
end

gray = mat2gray(gray);

bw = imbinarize(gray, 'adaptive', 'ForegroundPolarity', 'bright', 'Sensitivity', 0.45);
bw = bwareaopen(bw, 15);

if nnz(bw) < 20
    % Fallback threshold if adaptive thresholding under-segments.
    bw = gray > graythresh(gray) * 0.8;
    bw = bwareaopen(bw, 15);
end

stats = regionprops(bw, 'BoundingBox', 'Area');
if isempty(stats)
    error('No foreground digit was detected in the selected image.');
end

[~, idx] = max([stats.Area]);
bbox = stats(idx).BoundingBox;
crop = imcrop(gray, bbox);

% Pad to square, then resize to 20x20 and center in 28x28 canvas (MNIST style).
[h, w] = size(crop);
side = max(h, w);
padTop = floor((side - h) / 2);
padBottom = ceil((side - h) / 2);
padLeft = floor((side - w) / 2);
padRight = ceil((side - w) / 2);

cropSquare = padarray(crop, [padTop padLeft], 0, 'pre');
cropSquare = padarray(cropSquare, [padBottom padRight], 0, 'post');

core20 = imresize(cropSquare, [20 20], 'bilinear');
img28 = zeros(28, 28);
img28(5:24, 5:24) = core20;
img28 = mat2gray(img28);
end

function result = predict_with_all_models(img28, assets)
% Extract model-specific features and run all three classifiers.

rawFeature = reshape(img28.', 1, []);  % Row-major flattening to match CSV representation.
hogFeature = extractHOGFeatures(img28, 'CellSize', [4 4], 'NumBins', 9);
lbpFeature = extractLBPFeatures(img28);
statsFeature = extract_statistical_features_single(img28);
combinedFeature = [hogFeature, lbpFeature, statsFeature];

% SVM (ECOC)
[svmLabel, svmScore] = predict(assets.svm_model, hogFeature);
[svmProb, svmClasses] = score_to_probabilities(svmScore, assets.svm_model.ClassNames);
[result.svm.confidence, svmIdx] = max(svmProb);
result.svm.prediction = class_to_double(svmClasses(svmIdx));
if ~isempty(svmLabel)
    result.svm.prediction = class_to_double(svmLabel(1));
end

% Random Forest (TreeBagger)
[rfLabel, rfScore] = predict(assets.rf_model, combinedFeature);
rfProb = normalize_probabilities(rfScore(1, :));
[result.rf.confidence, rfIdx] = max(rfProb);
rfClasses = str2double(assets.rf_model.ClassNames);
result.rf.prediction = rfClasses(rfIdx);
if iscell(rfLabel)
    result.rf.prediction = str2double(rfLabel{1});
end

% k-NN
[knnLabel, knnScore] = predict(assets.knn_model, rawFeature);
knnProb = normalize_probabilities(knnScore(1, :));
[result.knn.confidence, knnIdx] = max(knnProb);
knnClasses = assets.knn_model.ClassNames;
result.knn.prediction = class_to_double(knnClasses(knnIdx));
if ~isempty(knnLabel)
    result.knn.prediction = class_to_double(knnLabel(1));
end
end

function stat_features = extract_statistical_features_single(img)
% Keep feature definition consistent with training in extract_features.m.

stat_features = zeros(1, 14);

stat_features(1) = mean(img(:));
stat_features(2) = std(img(:));
stat_features(3) = skewness(img(:));
stat_features(4) = kurtosis(img(:));

threshold = 0.5;
binary_img = img > threshold;
stat_features(5) = sum(binary_img(:)) / numel(img);

img_flip_h = fliplr(img);
img_flip_v = flipud(img);
stat_features(6) = corr2(img, img_flip_h);
stat_features(7) = corr2(img, img_flip_v);

h_proj = sum(img, 1);
v_proj = sum(img, 2);
stat_features(8) = std(h_proj);
stat_features(9) = std(v_proj);

[Gx, Gy] = gradient(img);
stat_features(10) = mean(abs(Gx(:)));
stat_features(11) = mean(abs(Gy(:)));

stat_features(12) = entropy(img);

edges = edge(img, 'Canny');
stat_features(13) = sum(edges(:)) / numel(edges);

[rows, cols] = find(binary_img);
if ~isempty(rows)
    height = max(rows) - min(rows) + 1;
    width = max(cols) - min(cols) + 1;
    stat_features(14) = height / max(width, 1);
else
    stat_features(14) = 1;
end

if any(~isfinite(stat_features))
    stat_features(~isfinite(stat_features)) = 0;
end
end

function [probabilities, classes] = score_to_probabilities(score, classNames)
% Convert model score vector to a normalized confidence vector.

classes = classNames;
if isempty(score)
    probabilities = zeros(1, numel(classes));
    return;
end

score = score(1, :);
if any(score < 0) || any(score > 1) || abs(sum(score) - 1) > 1e-3
    % For ECOC SVM raw scores, use softmax for relative confidence.
    shifted = score - max(score);
    expVals = exp(shifted);
    probabilities = expVals / sum(expVals + eps);
else
    probabilities = score;
end

probabilities = normalize_probabilities(probabilities);
end

function probs = normalize_probabilities(scores)
% Robust normalization helper for confidence display.

scores = double(scores(:)).';
scores(~isfinite(scores)) = 0;
scores(scores < 0) = 0;

total = sum(scores);
if total <= 0
    probs = zeros(size(scores));
    return;
end

probs = scores ./ total;
end

function value = class_to_double(c)
% Convert class label to numeric digit.

if iscell(c)
    c = c{1};
end

if isnumeric(c)
    value = double(c);
elseif iscategorical(c)
    value = str2double(string(c));
elseif isstring(c) || ischar(c)
    value = str2double(c);
else
    error('Unsupported class label type: %s', class(c));
end

if isnan(value)
    error('Could not convert predicted class to numeric digit.');
end
end

function rootDir = get_project_root()
% Resolve project root from script location.

thisFile = mfilename('fullpath');
scriptDir = fileparts(thisFile);
rootDir = fileparts(scriptDir);
end