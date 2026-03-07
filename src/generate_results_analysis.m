%% generate_results_analysis.m
% Compiles preprocessing, feature, training, and evaluation outputs into a
% single results analysis report (results/reports/results_analysis_report.txt).

function reportPath = generate_results_analysis()
% GENERATE_RESULTS_ANALYSIS  Produce results analysis from pipeline outputs.

rootDir = get_root_dir();
ensure_dirs(fullfile(rootDir, 'results', 'reports'));
reportPath = fullfile(rootDir, 'results', 'reports', 'results_analysis_report.txt');
fid = fopen(reportPath, 'w');
if fid == -1, error('Cannot create report: %s', reportPath); end

fprintf(fid, '================================================================================\n');
fprintf(fid, 'PATTERN RECOGNITION - RESULTS ANALYSIS\n');
fprintf(fid, 'MNIST Handwritten Digit Recognition (0-9)\n');
fprintf(fid, 'Generated: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, '================================================================================\n\n');

%% 1. Inputs / Data understanding
fprintf(fid, '1. INPUTS / DATA UNDERSTANDING\n');
fprintf(fid, '   - Datasets selected: MNIST (Modified National Institute of Standards and Technology).\n');
fprintf(fid, '   - Source: Kaggle CSV (mnist_train.csv, mnist_test.csv).\n');
fprintf(fid, '   - Training images: 60,000; Test images: 10,000.\n');
fprintf(fid, '   - Each image: 28x28 grayscale pixels (784 features), values 0-255.\n');
fprintf(fid, '   - Classes: 10 (digits 0-9). Label distribution is approximately balanced.\n\n');

%% 2. Preprocessing justifications
fprintf(fid, '2. PREPROCESSING JUSTIFICATIONS\n');
prepPath = fullfile(rootDir, 'results', 'preprocessed', 'train_val_test.mat');
if exist(prepPath, 'file')
    d = load(prepPath);
    fprintf(fid, '   - Number of samples: Training %d, Validation %d, Test %d.\n', ...
        d.n_train_final, d.n_val_final, d.n_test);
    fprintf(fid, '   - Number of features (before feature extraction): %d.\n', d.n_features_raw);
    fprintf(fid, '   - Number of classes: %d.\n', d.n_classes);
else
    fprintf(fid, '   - (Run pipeline to fill: training/validation/test counts, features, classes.)\n');
end
fprintf(fid, '   - Normalization: min-max [0,255] -> [0,1] for numerical stability and classifier performance.\n');
fprintf(fid, '   - Train/validation split: 85%% train, 15%% validation (seed=42). Test set kept separate.\n\n');

%% 3. Features selection / engineering
fprintf(fid, '3. FEATURES SELECTION / ENGINEERING\n');
fprintf(fid, '   - Method selected: (a) Raw normalized pixels (784-dim), (b) PCA (100 components).\n');
fprintf(fid, '   - Why: Raw pixels as baseline; PCA reduces dimensionality, retains most variance,\n');
fprintf(fid, '     speeds training and can reduce overfitting. PCA is linear and well-suited to\n');
fprintf(fid, '     correlated pixel data in digit images.\n');
featPath = fullfile(rootDir, 'results', 'features', 'features_pca.mat');
if exist(featPath, 'file')
    d = load(featPath);
    var_ret = sum(d.explained(1:min(d.n_components, numel(d.explained))));
    fprintf(fid, '   - Variance retained by PCA: %.2f%%.\n', var_ret);
end
fprintf(fid, '\n');

%% 4. Challenges and how handled
fprintf(fid, '4. CHALLENGES AND HOW THEY ARE HANDLED\n');
fprintf(fid, '   - High dimensionality (784): Addressed by PCA and by using efficient classifiers.\n');
fprintf(fid, '   - Long training time (e.g. Random Forest): Used PCA features and reduced tree count\n');
fprintf(fid, '     or MinLeafSize for faster runs; full hyperparameter search can be run separately.\n');
fprintf(fid, '   - Multiclass (10 classes): SVM via one-vs-one (fitcecoc); others natively multiclass.\n');
fprintf(fid, '   - Reproducibility: Fixed random seed (rng(42)) for splits and training.\n\n');

%% 5. Outputs
fprintf(fid, '5. OUTPUTS\n');
fprintf(fid, '   - Preprocessed data: results/preprocessed/train_val_test.mat\n');
fprintf(fid, '   - Features: results/features/features_raw.mat, features_pca.mat\n');
fprintf(fid, '   - Trained models: results/models/trained_models.mat (Naive Bayes, k-NN, LDA, RF, SVM)\n');
fprintf(fid, '   - Reports: results/reports/preprocessing_report.txt, feature_extraction_report.txt,\n');
fprintf(fid, '     training_report.txt, evaluation_report.txt, results_analysis_report.txt.\n');
fprintf(fid, '   - Test-set metrics: accuracy, precision, recall, specificity, confusion matrices.\n\n');

%% 6. Optimization / fine tuning
fprintf(fid, '6. OPTIMIZATION / FINE TUNING\n');
fprintf(fid, '   - PCA: Number of components (e.g. 50, 100) can be tuned via validation accuracy.\n');
fprintf(fid, '   - k-NN: k tuned (e.g. k=5); distance metric (Euclidean) and Standardize=false for normalized data.\n');
fprintf(fid, '   - Random Forest: NumTrees (100 default; increase for better accuracy), MinLeafSize.\n');
fprintf(fid, '   - SVM: Kernel (linear used for speed); C and kernel scale can be cross-validated.\n');
fprintf(fid, '   - Validation set used for model selection and early comparison; final metrics on test set.\n\n');

%% 7. Critical analysis
fprintf(fid, '7. CRITICAL ANALYSIS\n');
metPath = fullfile(rootDir, 'results', 'reports', 'evaluation_metrics.mat');
if exist(metPath, 'file')
    m = load(metPath);
    metrics = m.metrics;
    names = {'NaiveBayes', 'kNN', 'LDA', 'RandomForest', 'SVM'};
    accs = [];
    for i = 1:length(names)
        if isfield(metrics, names{i})
            accs(i) = metrics.(names{i}).accuracy;
        else
            accs(i) = NaN;
        end
    end
    [best_acc, idx] = max(accs);
    best_name = names{idx};
    fprintf(fid, '   - Best model on test set: %s (accuracy %.4f).\n', best_name, best_acc);
    fprintf(fid, '   - Why this model may be better: Task-dependent; SVM and k-NN often perform well on\n');
    fprintf(fid, '     image features; LDA is fast and interpretable; RF is robust to hyperparameters.\n');
    fprintf(fid, '   - Feature extraction: PCA reduces noise and redundancy; comparison with raw features\n');
    fprintf(fid, '     can be done by training on features_raw.mat for selected models.\n');
    fprintf(fid, '   - Limitations: (1) MNIST is relatively easy; real handwriting may vary more. (2) Linear\n');
    fprintf(fid, '     PCA may miss non-linear structure. (3) Deployment must replicate same preprocessing\n');
    fprintf(fid, '     and PCA transform (saved coeff, mu) for new images.\n');
else
    fprintf(fid, '   - (Run evaluate_model.m and re-run pipeline to fill critical analysis with test metrics.)\n');
end

fprintf(fid, '\n================================================================================\n');
fprintf(fid, 'END OF RESULTS ANALYSIS\n');
fclose(fid);
fprintf('Results analysis written to: %s\n', reportPath);
end

function ensure_dirs(varargin)
for i = 1:nargin
    if ~exist(varargin{i}, 'dir'), mkdir(varargin{i}); end
end
end

function rootDir = get_root_dir()
[cwd, name, ~] = fileparts(pwd);
if strcmp(name, 'src'), rootDir = cwd; else, rootDir = pwd; end
end
