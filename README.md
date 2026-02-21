# MNIST Digit Classifier

**Course:** CT104-3-M Pattern Recognition  
**Institution:** Asia Pacific University of Technology & Innovation (APU)  
**Assignment:** Handwritten Digit Recognition System

Pattern Recognition assignment for classifying handwritten digits (0-9) using the MNIST dataset.

## Team Members
- LOH HOI PING
- TEE MUN CHUN
- ROHAN MAZUMDAR

## Project Structure (Draft - Subject to Change as per team discussion)
```
mnist-digit-classifier/
├── data/              # Dataset files (CSV and .mat)
├── src/               # Source code
├── results/           # Output results
└── README.md
```

## Planned Pipeline Structure (Draft - Subject to Change as per team discussion)
```
src/
├── load_data.m              # Load and explore MNIST dataset
├── preprocess_data.m        # Normalize and clean data (TODO)
├── extract_features.m       # Feature extraction: PCA, HOG, etc. (TODO)
├── train_model.m            # Train classifiers (k-NN, SVM, etc.) (TODO)
├── evaluate_model.m         # Performance metrics and analysis (TODO)
└── main_pipeline.m          # Run complete workflow (TODO)
```

**Workflow:** Each script processes data and saves output for the next step.

## How to Run Locally

### Prerequisites
- MATLAB installed on your system OR Install MATLAB extension (by MathWorks) in VSCODE
- Git installed

### Setup Steps

1. **Clone the repository**
    ```bash
    git clone https://github.com/YOUR_USERNAME/mnist-digit-classifier.git

    cd mnist-digit-classifier
    ```

2. **Run in MATLAB**
   - Open MATLAB
   - Navigate to the project root folder: `mnist-digit-classifier`
   - Run the data loading script:
        ```matlab
        run('src/load_data.m')
        ```

3. **Run in VSCode:**
   - Open project folder in VSCode
   - Install MATLAB extension (by MathWorks)
   - Open `src/load_data.m`
   - Press `F5` or click the Run the PLAY button on the top-right corner
   - Make sure your current directory in MATLAB is the project root
   - to set project root
        ```bash
        cd('/<path-to>/mnist-digit-classifier')
        ```


## Project Goal
Build a machine learning system to recognize handwritten digits with high accuracy.

## Status
Work in progress

## Tech Stack
- MATLAB
- MNIST Dataset
