%% evaluate_model.m
% Testing and Evaluation (1.8)
% Loads trained models and test set; computes accuracy, precision, recall,
% specificity, confusion matrix; writes results to report for documentation.

function [metrics, reportPath] = evaluate_model(varargin)
% EVALUATE_MODEL  Evaluate all saved models on test set and write report.
% Always writes results/reports/evaluation_report.txt and evaluation_metrics.mat.

rootDir = get_root_dir();
ensure_dirs(fullfile(rootDir, 'results', 'reports'));

modelPath = fullfile(rootDir, 'results', 'models', 'trained_models.mat');
featPath  = fullfile(rootDir, 'results', 'features', 'features_pca.mat');
if ~exist(modelPath, 'file'), error('Run train_model.m first.'); end
if ~exist(featPath, 'file'),  error('Run extract_features.m first.'); end

data_m = load(modelPath);
data_f = load(featPath);
models = data_m.models;
X_test = data_f.X_test_pca;
test_Y = double(data_f.test_Y);  % numeric 0-9 for metrics
test_Y_cat = categorical(test_Y);
class_list = (0:9)';
n_classes = length(class_list);

reportPath = fullfile(rootDir, 'results', 'reports', 'evaluation_report.txt');
fid = fopen(reportPath, 'w');
if fid == -1, error('Cannot create report: %s', reportPath); end
fprintf(fid, '================================================================================\n');
fprintf(fid, 'EVALUATION REPORT - MNIST Handwritten Digit Recognition (TEST SET)\n');
fprintf(fid, 'Generated: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'Test samples: %d | Classes: 0-9\n', size(X_test, 1));
fprintf(fid, '================================================================================\n\n');

metrics = struct();

model_names = {'NaiveBayes', 'kNN', 'LDA', 'RandomForest', 'SVM'};
for m = 1:length(model_names)
    name = model_names{m};
    if ~isfield(models, name), continue; end
    mdl = models.(name);

    % Predict (handle TreeBagger differently)
    if strcmp(name, 'RandomForest')
        pred_cell = mdl.predict(X_test);
        pred = str2double(string(pred_cell));
    else
        pred_cat = predict(mdl, X_test);
        pred = double(pred_cat);
    end

    % Clip to valid classes (in case of any NaN)
    pred = max(0, min(9, round(pred)));

    % Accuracy
    acc = sum(pred == test_Y) / numel(test_Y);

    % Per-class: TP, FP, FN, TN for one-vs-rest → Precision, Recall (sensitivity), Specificity
    precisions = zeros(n_classes, 1);
    recalls    = zeros(n_classes, 1);
    specificities = zeros(n_classes, 1);
    for c = 1:n_classes
        cls = class_list(c);
        TP = sum((pred == cls) & (test_Y == cls));
        FP = sum((pred == cls) & (test_Y ~= cls));
        FN = sum((pred ~= cls) & (test_Y == cls));
        TN = sum((pred ~= cls) & (test_Y ~= cls));
        precisions(c) = TP / (TP + FP + 1e-10);
        recalls(c)    = TP / (TP + FN + 1e-10);  % sensitivity
        specificities(c) = TN / (TN + FP + 1e-10);
    end
    macro_precision = mean(precisions);
    macro_recall    = mean(recalls);
    macro_specificity = mean(specificities);

    % Confusion matrix (rows = true, cols = pred)
    C = confusionmat(test_Y, pred, 'Order', 0:9);

    metrics.(name).accuracy = acc;
    metrics.(name).precision_macro = macro_precision;
    metrics.(name).recall_macro = macro_recall;
    metrics.(name).specificity_macro = macro_specificity;
    metrics.(name).confusion_matrix = C;
    metrics.(name).precisions = precisions;
    metrics.(name).recalls = recalls;

    fprintf(fid, '--------------------------------------------------------------------------------\n');
    fprintf(fid, 'Model: %s\n', name);
    fprintf(fid, '--------------------------------------------------------------------------------\n');
    fprintf(fid, '  Accuracy:        %.4f\n', acc);
    fprintf(fid, '  Macro Precision: %.4f\n', macro_precision);
    fprintf(fid, '  Macro Recall (Sensitivity): %.4f\n', macro_recall);
    fprintf(fid, '  Macro Specificity: %.4f\n', macro_specificity);
    fprintf(fid, '  Confusion Matrix (rows=true, cols=predicted):\n');
    fmt = [repmat(' %6d', 1, 10) '\n'];
    fprintf(fid, '    ');
    fprintf(fid, fmt, 0:9);
    for r = 1:10
        fprintf(fid, ' %2d ', class_list(r));
        fprintf(fid, fmt, C(r,:));
    end
    fprintf(fid, '\n');
end

fprintf(fid, '================================================================================\n');
fprintf(fid, 'END OF EVALUATION REPORT\n');
fclose(fid);
fprintf('Evaluation report written to: %s\n', reportPath);
save(fullfile(rootDir, 'results', 'reports', 'evaluation_metrics.mat'), 'metrics');
end

function rootDir = get_root_dir()
[cwd, name, ~] = fileparts(pwd);
if strcmp(name, 'src'), rootDir = cwd; else, rootDir = pwd; end
end

function ensure_dirs(varargin)
for i = 1:nargin
    if ~exist(varargin{i}, 'dir'), mkdir(varargin{i}); end
end
end
