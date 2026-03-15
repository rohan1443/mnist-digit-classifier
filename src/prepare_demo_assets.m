function prepare_demo_assets(samplesPerDigit)
% prepare_demo_assets.m
% Export representative MNIST test images into PNG/JPG/GIF for demo and submission.
%
% Usage:
%   prepare_demo_assets();       % default 3 samples per digit
%   prepare_demo_assets(5);      % 5 samples per digit

if nargin < 1
    samplesPerDigit = 3;
end

if samplesPerDigit < 1
    error('samplesPerDigit must be >= 1.');
end

rootDir = get_project_root();
csvPath = fullfile(rootDir, 'data', 'csv', 'mnist_test.csv');
outDir = fullfile(rootDir, 'demo', 'test_images');

if ~isfile(csvPath)
    error('MNIST test CSV not found: %s', csvPath);
end

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

fprintf('Loading test CSV from: %s\n', csvPath);
M = readmatrix(csvPath);
labels = M(:, 1);
pixels = M(:, 2:end);

manifestPath = fullfile(outDir, 'manifest.csv');
fid = fopen(manifestPath, 'w');
fprintf(fid, 'filename,label,format\n');

totalWritten = 0;
rng(42);

for digit = 0:9
    idx = find(labels == digit);
    if isempty(idx)
        continue;
    end

    pickCount = min(samplesPerDigit, numel(idx));
    pick = idx(randperm(numel(idx), pickCount));

    for j = 1:pickCount
        row = pixels(pick(j), :);
        img = reshape(row, 28, 28)';
        img = uint8(img);

        imgBig = imresize(img, [280 280], 'nearest');

        pngName = sprintf('digit_%d_%02d.png', digit, j);
        jpgName = sprintf('digit_%d_%02d.jpg', digit, j);
        gifName = sprintf('digit_%d_%02d.gif', digit, j);

        imwrite(imgBig, fullfile(outDir, pngName));
        imwrite(imgBig, fullfile(outDir, jpgName), 'Quality', 95);
        imwrite(imgBig, fullfile(outDir, gifName));

        fprintf(fid, '%s,%d,png\n', pngName, digit);
        fprintf(fid, '%s,%d,jpg\n', jpgName, digit);
        fprintf(fid, '%s,%d,gif\n', gifName, digit);

        totalWritten = totalWritten + 3;
    end
end

fclose(fid);

fprintf('Generated %d image files in %s\n', totalWritten, outDir);
fprintf('Manifest: %s\n', manifestPath);
end

function rootDir = get_project_root()
thisFile = mfilename('fullpath');
scriptDir = fileparts(thisFile);
rootDir = fileparts(scriptDir);
end