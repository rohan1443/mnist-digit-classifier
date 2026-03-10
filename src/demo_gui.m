%% demo_gui.m
% Simple GUI for MNIST Digit Recognition Demo
%
% This creates a user-friendly interface to:
% 1. Upload a digit image
% 2. Preprocess it automatically
% 3. Predict using the best trained model
% 4. Display result with confidence
%
% Usage: Run this script to launch the GUI
%        gui = demo_gui;

function demo_gui()
    % Main function to create and launch the GUI
    
    %% Load best trained model
    fprintf('Loading best model...\n');
    
    % Load evaluation results to find best model
    load('results/evaluate_model/evaluation_metrics.mat', 'best_f', 'best_m', 'feature_names', 'model_names');
    
    % Load the best model file
    if best_m == 1 % k-NN
        model_file = sprintf('results/models/knn_%s.mat', lower(feature_names{best_f}));
        model_data = load(model_file);
        trained_model = model_data.knn_final;
        model_type = 'knn';
    elseif best_m == 2 % SVM
        model_file = sprintf('results/models/svm_%s.mat', lower(feature_names{best_f}));
        model_data = load(model_file);
        trained_model = model_data.svm_model;
        model_type = 'svm';
    else % Random Forest
        model_file = sprintf('results/models/rf_%s.mat', lower(feature_names{best_f}));
        model_data = load(model_file);
        trained_model = model_data.rf_final;
        model_type = 'rf';
    end
    
    best_feature = feature_names{best_f};
    best_model = model_names{best_m};
    
    % Load feature extraction coefficients if needed
    load('data/feature-extracted/features.mat', 'pca50_coeff', 'pca100_coeff', 'hybrid_coeff');
    
    % Load preprocessing parameters
    load('data/preprocessed/mnist_preprocessed.mat', 'mean_pixel');
    
    fprintf('Loaded: %s with %s features\n\n', best_model, best_feature);
    
    %% Create GUI Figure
    % Create main window
    fig = figure('Name', 'MNIST Digit Recognizer - Demo', ...
                 'Position', [100, 100, 800, 600], ... % [left, bottom, width, height]
                 'MenuBar', 'none', ... % Remove default menu
                 'NumberTitle', 'off', ... % Remove "Figure 1" title
                 'Resize', 'off'); % Fixed size window
    
    % Set background color (light gray)
    set(fig, 'Color', [0.94 0.94 0.94]);
    
    %% Title Text
    uicontrol('Parent', fig, ...
              'Style', 'text', ... % Text label
              'String', 'MNIST Handwritten Digit Recognition', ...
              'FontSize', 18, ...
              'FontWeight', 'bold', ...
              'BackgroundColor', [0.94 0.94 0.94], ...
              'Position', [50, 530, 700, 40]); % [left, bottom, width, height]
    
    % Subtitle with model info
    uicontrol('Parent', fig, ...
              'Style', 'text', ...
              'String', sprintf('Using: %s with %s Features', best_model, best_feature), ...
              'FontSize', 12, ...
              'BackgroundColor', [0.94 0.94 0.94], ...
              'Position', [50, 500, 700, 25]);
    
    %% Image Display Area
    % Create axes for showing uploaded image
    img_axes = axes('Parent', fig, ...
                    'Units', 'pixels', ...
                    'Position', [100, 250, 250, 250]); % Square display area
    axis(img_axes, 'off'); % Hide axis lines
    title(img_axes, 'Upload an Image', 'FontSize', 14);
    
    %% Upload Button
    upload_btn = uicontrol('Parent', fig, ...
                          'Style', 'pushbutton', ... % Button
                          'String', 'Upload Digit Image', ...
                          'FontSize', 12, ...
                          'Position', [100, 190, 200, 40], ...
                          'Callback', @upload_callback); 
                          % Callback: function to run when button clicked
    
    %% Recognize Button (initially disabled)
    recognize_btn = uicontrol('Parent', fig, ...
                             'Style', 'pushbutton', ...
                             'String', 'Recognize Digit', ...
                             'FontSize', 14, ...
                             'FontWeight', 'bold', ...
                             'Position', [100, 140, 200, 50], ...
                             'Enable', 'off', ... % Disabled until image uploaded
                             'BackgroundColor', [0.2 0.6 0.9], ...
                             'ForegroundColor', 'white', ...
                             'Callback', @recognize_callback);
    
    %% Result Display Area
    result_axes = axes('Parent', fig, ...
                      'Units', 'pixels', ...
                      'Position', [450, 250, 300, 250]);
    axis(result_axes, 'off');
    title(result_axes, 'Prediction Result', 'FontSize', 14);
    
    % Text to show prediction
    result_text = uicontrol('Parent', fig, ...
                           'Style', 'text', ...
                           'String', 'No prediction yet', ...
                           'FontSize', 16, ...
                           'FontWeight', 'bold', ...
                           'BackgroundColor', [0.94 0.94 0.94], ...
                           'ForegroundColor', [0.2 0.2 0.2], ...
                           'Position', [450, 350, 300, 80]);
    
    % Text to show confidence
    confidence_text = uicontrol('Parent', fig, ...
                               'Style', 'text', ...
                               'String', '', ...
                               'FontSize', 16, ...
                               'BackgroundColor', [0.94 0.94 0.94], ...
                               'ForegroundColor', [0.4 0.4 0.4], ...
                               'Position', [450, 310, 300, 30]);
    
    %% Probability Bar Chart Area (for top 3 predictions)
    prob_axes = axes('Parent', fig, ...
                    'Units', 'pixels', ...
                    'Position', [450, 140, 300, 150]);
    title(prob_axes, 'Prediction Probabilities', 'FontSize', 12);
    
    %% Instructions Text
    instructions = {
        'Instructions:', ...
        '1. Click "Upload Digit Image" to select an image', ...
        '2. Image should be 28x28 grayscale or will be converted', ...
        '3. Click "Recognize Digit" to predict', ...
        '4. View prediction and confidence score'
    };
    
    uicontrol('Parent', fig, ...
              'Style', 'text', ...
              'String', instructions, ...
              'FontSize', 10, ...
              'HorizontalAlignment', 'left', ...
              'BackgroundColor', [0.94 0.94 0.94], ...
              'Position', [50, 20, 700, 100]);
    
    %% Variables to store current image
    current_image = [];
    current_features = [];
    
    %% ========================================
    %% Callback: Upload Image
    %% ========================================
    function upload_callback(~, ~)
        % This function runs when "Upload Image" button is clicked
        
        % Open file dialog to select image
        [filename, filepath] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp;*.gif', 'Image Files'}, ...
                                         'Select a digit image');
        % uigetfile: opens file browser dialog
        
        if filename == 0
            % User cancelled
            return;
        end
        
        % Load image
        full_path = fullfile(filepath, filename);
        img = imread(full_path); % imread: reads image file
        
        % Convert to grayscale if needed
        if size(img, 3) == 3
            img = rgb2gray(img); % rgb2gray: converts color to grayscale
        end
        
        % Resize to 28x28 if needed
        if size(img, 1) ~= 28 || size(img, 2) ~= 28
            img = imresize(img, [28, 28]); % imresize: changes image size
        end
        
        % Convert to double and normalize to [0, 1]
        img = double(img) / 255.0;
        
        % Store for later use
        current_image = img;
        
        % Display uploaded image
        axes(img_axes); % Switch to image axes
        imshow(img, []);
        title('Uploaded Image', 'FontSize', 14);
        
        % Preprocess and extract features
        current_features = extract_features_for_gui(img);
        
        % Enable recognize button
        set(recognize_btn, 'Enable', 'on');
        
        % Clear previous results
        set(result_text, 'String', 'Ready to predict');
        set(confidence_text, 'String', '');
        cla(prob_axes); % Clear probability plot
        title(prob_axes, 'Prediction Probabilities', 'FontSize', 12);
        
        fprintf('Image loaded and preprocessed.\n');
    end
    
    %% ========================================
    %% Callback: Recognize Digit
    %% ========================================
    function recognize_callback(~, ~)
        % This function runs when "Recognize Digit" button is clicked
        
        if isempty(current_features)
            errordlg('Please upload an image first!', 'No Image');
            % errordlg: shows error dialog box
            return;
        end
        
        fprintf('Predicting...\n');
        
        % Predict using trained model
        if strcmp(model_type, 'rf')
            % Random Forest returns cell array
            pred_cell = predict(trained_model, current_features);
            prediction = str2double(pred_cell{1});
            
            % Get probabilities (scores from each tree)
            % For Random Forest, approximate confidence
            confidence = 0.95; % Placeholder - RF doesn't give direct probabilities easily
            probs = zeros(1, 10);
            probs(prediction + 1) = confidence; % Simple approximation
            
        else
            % k-NN or SVM
            [prediction, scores] = predict(trained_model, current_features);
            % scores: decision values or distances
            
            % Convert scores to probabilities (approximate)
            if strcmp(model_type, 'knn')
                % For k-NN, high confidence if neighbors agree
                confidence = 0.95; % Simplified
                probs = zeros(1, 10);
                probs(prediction + 1) = confidence;
            else
                % For SVM, convert scores to probabilities
                % Simple softmax approximation
                exp_scores = exp(scores - max(scores)); % Prevent overflow
                probs = exp_scores / sum(exp_scores);
                confidence = probs(prediction + 1);
            end
        end
        
        fprintf('Predicted: %d (Confidence: %.2f%%)\n', prediction, confidence * 100);
        
        % Display prediction
        set(result_text, 'String', sprintf('%d', prediction), ...
            'ForegroundColor', [0.1 0.6 0.1]); % Green color
        
        set(confidence_text, 'String', sprintf('Confidence: %.1f%%', confidence * 100));
        
        % Show top 3 predictions with probabilities
        [sorted_probs, sorted_idx] = sort(probs, 'descend');
        top_3_digits = sorted_idx(1:3) - 1; % Convert to 0-9
        top_3_probs = sorted_probs(1:3);
        
        % Plot probability bars
        axes(prob_axes);
        bar(top_3_digits, top_3_probs * 100);
        xlabel('Digit');
        ylabel('Probability (%)');
        title('Top 3 Predictions');
        ylim([0 100]);
        xticks(top_3_digits);
        grid on;
        
        % Highlight predicted digit in image
        axes(result_axes);
        imshow(current_image, []);
        title(sprintf('Predicted: %d', prediction), 'FontSize', 16, 'FontWeight', 'bold');
        
    end
    
    %% ========================================
    %% Helper: Extract Features for GUI
    %% ========================================
    function features = extract_features_for_gui(img)
        % Extract features based on best feature type
        
        % Flatten image to vector (28x28 -> 784)
        img_vector = reshape(img', 1, 784); % Note: transpose for correct orientation
        
        % Extract features based on best feature type
        if strcmp(best_feature, 'Raw')
            % Raw pixels - just use normalized vector
            features = img_vector;
            
        elseif strcmp(best_feature, 'PCA50')
            % Apply PCA transformation with 50 components
            features = img_vector * pca50_coeff;
            
        elseif strcmp(best_feature, 'PCA100')
            % Apply PCA transformation with 100 components
            features = img_vector * pca100_coeff;
            
        elseif strcmp(best_feature, 'HOG')
            % Extract HOG features
            img_2d = reshape(img_vector, 28, 28)';
            features = extractHOGFeatures(img_2d, 'CellSize', [4 4]);
            
        elseif strcmp(best_feature, 'Hybrid')
            % Extract HOG then apply PCA
            img_2d = reshape(img_vector, 28, 28)';
            hog_features = extractHOGFeatures(img_2d, 'CellSize', [4 4]);
            features = hog_features * hybrid_coeff;
        end
        
    end
    
    fprintf('\nGUI launched successfully!\n');
    fprintf('Upload an image to begin recognition.\n\n');
    
end