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

## Pipeline Structure
```
src/
├── load_data.m                  # Load and explore MNIST dataset (data understanding)
├── preprocess_data.m            # Normalize, split train/val/test (1.3, 1.4)
├── extract_features.m           # Raw + PCA features (1.5)
├── train_model.m                # Naive Bayes, k-NN, LDA, Random Forest, SVM (1.6, 1.7)
├── evaluate_model.m             # Accuracy, precision, recall, confusion matrix (1.8)
├── generate_results_analysis.m  # Compiles results analysis report from all steps
└── main_pipeline.m              # Run full workflow and generate all reports
```

**Workflow:** Each step saves outputs to `results/` and writes a report to `results/reports/`. The pipeline always generates all reports so you capture results once (no need to re-run for evidence; training can take a long time).

---

## Running in Cursor (or VSCode)

The project runs in **Cursor** (and VSCode) using the **MATLAB extension**. You need **MATLAB installed** on your machine; the extension connects to it to run `.m` files.

### Adding the MATLAB extension in Cursor

1. **Open Cursor** and open this project folder (`mnist-digit-classifier`).
2. **Open the Extensions view:**
   - Shortcut: `Ctrl+Shift+X` (Windows/Linux) or `Cmd+Shift+X` (Mac)
   - Or click the Extensions icon in the left sidebar.
3. **Search for MATLAB:**
   - In the search box, type: `MATLAB`
   - Find **"MATLAB"** by **MathWorks** (publisher: MathWorks).
4. **Install:**
   - Click **Install** on the MATLAB extension.
   - Wait until it finishes installing.
5. **Connect to MATLAB:**
   - You must have **MATLAB** installed (e.g. from your university or [MathWorks](https://www.mathworks.com/products/matlab.html)).
   - The extension runs your code using that installation. You **do not** run `.m` files from the terminal; you run them from the editor (e.g. **F5**). If the terminal says `matlab: command not found`, that only means MATLAB is not on your shell PATH—the extension can still work if it finds MATLAB (see step 6).
6. **(Optional) Set MATLAB path in settings:**
   - Open Settings: `Ctrl+,` (Windows/Linux) or `Cmd+,` (Mac)
   - Search for `matlab.matlabpath`
   - Set it to your MATLAB executable folder if the extension does not find it (e.g. `C:\Program Files\MATLAB\R2024a\bin` on Windows or `/Applications/MATLAB_R2024a.app/bin` on Mac).

### Running the pipeline in Cursor

1. **Set the project root as the current folder in MATLAB:**
   - Open the **Command Palette:** `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac).
   - Run: **"MATLAB: Change Folder"** (or **"MATLAB: Change to Project Root"** if available).
   - Choose the `mnist-digit-classifier` folder (the one that contains `src` and `data`).
   - Or in the MATLAB Command Window (if the extension opens it), run:
     ```matlab
     cd('/full/path/to/mnist-digit-classifier')
     ```
2. **Run the full pipeline:**
   - Open `src/main_pipeline.m` in the editor.
   - Press **F5** or click **Run** (play button) in the editor toolbar.
   - The extension will run the script in MATLAB. Ensure the **current folder** is the project root (see step 1).
3. **If you see "current folder" or path errors:**
   - In the MATLAB Command Window, run: `cd('/full/path/to/mnist-digit-classifier')` with your actual path, then run `main_pipeline` or run `main_pipeline.m` again from Cursor.

---

## Running in MATLAB desktop (or other editors)

If you use **MATLAB desktop** instead of Cursor, or any editor that can run MATLAB:

### Prerequisites
- **MATLAB** installed (Statistics and Machine Learning Toolbox needed)
- **Git** installed
- **MNIST data:** place `mnist_train.csv` and `mnist_test.csv` in `data/csv/` (from [Kaggle MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data))

---

## Step-by-step: End-to-end execution

Follow these steps in order. Works in **Cursor (with MATLAB extension)**, **VSCode (with MATLAB extension)**, or **MATLAB desktop**.

### Step 1: Get the project and data
1. Clone the repository and go into the project folder:
   ```bash
   git clone https://github.com/YOUR_USERNAME/mnist-digit-classifier.git
   cd mnist-digit-classifier
   ```
2. Create the data folder and add the MNIST CSV files:
   - Create folder: `data/csv/`
   - Download from Kaggle: [MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data)
   - Put `mnist_train.csv` and `mnist_test.csv` inside `data/csv/`
   - You should have: `mnist-digit-classifier/data/csv/mnist_train.csv` and `mnist-digit-classifier/data/csv/mnist_test.csv`

### Step 2: Set project root
- **In Cursor/VSCode (MATLAB extension):** Use **MATLAB: Change Folder** to the `mnist-digit-classifier` folder, or in the MATLAB command line run `cd('/full/path/to/mnist-digit-classifier')`.
- **In MATLAB desktop:** In the Command Window run:
  ```matlab
  cd('/full/path/to/mnist-digit-classifier')
  ```
  Example Mac: `cd('/Users/yourname/Documents/GitHub/mnist-digit-classifier')`  
  Example Windows: `cd('C:\Users\yourname\Documents\mnist-digit-classifier')`
- Confirm: `pwd` and `dir` should show `data`, `src`, `README.md`, etc.

### Step 3: (Optional) Run data understanding only
- To only load and inspect the dataset (no training):
  ```matlab
  run('src/load_data.m')
  ```
- You can skip this and go straight to the full pipeline.

### Step 4: Run the full pipeline
1. With current folder = **project root**, run one of these:
   - **Cursor or VSCode (recommended if you use the MATLAB extension):** Open `src/main_pipeline.m` and press **F5** (or click Run). No terminal command needed—the extension runs MATLAB for you.
   - **From terminal:** Only if MATLAB is on your PATH (or in `/Applications/MATLAB_*.app/bin` on Mac), run `./run_pipeline.sh`. If you see `matlab: command not found`, use **F5** in Cursor instead.
   - **MATLAB desktop:** In the Command Window run:
     ```matlab
     main_pipeline
     ```
     Or: `run('src/main_pipeline.m')`
2. The pipeline will:
   - **Step 1:** Load CSV → normalize → split train/val/test → save `results/preprocessed/train_val_test.mat` and `preprocessing_report.txt`
   - **Step 2:** Extract raw + PCA features → save `results/features/*.mat` and `feature_extraction_report.txt`
   - **Step 3:** Train Naive Bayes, k-NN, LDA, Random Forest, SVM → save `results/models/trained_models.mat` and `training_report.txt` (training can take several minutes)
   - **Step 4:** Evaluate all models on the test set → `evaluation_report.txt` and `evaluation_metrics.mat`
   - **Step 5:** Generate **results analysis** → `results/reports/results_analysis_report.txt`
3. When it finishes you should see: `========== Pipeline complete ==========`

### Step 5: Python deployment (optional)
1. Install Python dependencies (from project root):
   ```bash
   pip install -r python/requirements.txt
   ```
2. Train the Python model (same pipeline as MATLAB):
   ```bash
   python python/train_pipeline.py
   ```
   Requires `data/csv/mnist_train.csv` and `mnist_test.csv`.
3. Start the web app:
   ```bash
   streamlit run python/app.py
   ```
4. Open the URL shown in the terminal, upload a handwritten digit image, and get a prediction.

---

### Quick reference: run order
| Order | Action | Command (from project root) |
|-------|--------|-----------------------------|
| 1 | Set project root | In MATLAB: `cd('/path/to/mnist-digit-classifier')` or use MATLAB: Change Folder in Cursor |
| 2 | Full pipeline (terminal) | `./run_pipeline.sh` (requires MATLAB on PATH) |
| 2 | Full pipeline (MATLAB / Cursor) | `main_pipeline` or open `src/main_pipeline.m` and press F5 |
| 3 | (Optional) Data only | In MATLAB: `run('src/load_data.m')` |

---

## Project Goal
Build a machine learning system to recognize handwritten digits with high accuracy.

## Results analysis (reports)

After running `main_pipeline`, all reports are written to `results/reports/`. Use them as evidence in your assignment:

- **Summary:** `results/reports/results_analysis_report.txt` (compiled analysis: inputs, preprocessing, features, challenges, outputs, optimization, critical analysis)
- **1. Inputs / Data understanding:** `preprocessing_report.txt` and section 1 of `results_analysis_report.txt`
- **2. Preprocessing justifications:** `preprocessing_report.txt` (samples, features, classes)
- **3. Features selection/engineering:** `feature_extraction_report.txt`
- **4. Challenges and handling:** Section 4 of `results_analysis_report.txt`
- **5. Outputs:** All files under `results/` and the reports listed above
- **6. Optimization/fine tuning:** Section 6 of `results_analysis_report.txt`; `training_report.txt`
- **7. Critical analysis:** Section 7 of `results_analysis_report.txt`; `evaluation_report.txt`

## Status
Implementation complete; ready for training runs and report writing.

## Tech Stack
- MATLAB (Statistics and Machine Learning Toolbox)
- MNIST Dataset (Kaggle CSV)
- Python (optional): scikit-learn, Streamlit for deployment UI
