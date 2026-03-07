"""
Python training pipeline: replicates MATLAB preprocessing, PCA, and best model (SVM)
for deployment. Saves model and PCA so the Streamlit app can run without MATLAB.
Run from project root: python python/train_pipeline.py
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Project root (parent of python/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "csv")
OUT_DIR = os.path.join(ROOT, "python", "models")
REPORT_DIR = os.path.join(ROOT, "results", "reports")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_PCA_COMPONENTS = 100  # match MATLAB


def load_mnist():
    """Load MNIST from CSV (same as MATLAB)."""
    train = pd.read_csv(os.path.join(DATA_DIR, "mnist_train.csv"), header=None)
    test = pd.read_csv(os.path.join(DATA_DIR, "mnist_test.csv"), header=None)
    y_train = train.iloc[:, 0].values
    X_train = train.iloc[:, 1:].values.astype(np.float64)
    y_test = test.iloc[:, 0].values
    X_test = test.iloc[:, 1:].values.astype(np.float64)
    return X_train, y_train, X_test, y_test


def normalize(X):
    """Min-max to [0, 1] (match MATLAB)."""
    return (X - 0) / 255.0


def split_validation(X_train, y_train, val_frac=0.15):
    """85% train, 15% validation."""
    n = len(X_train)
    np.random.seed(RANDOM_STATE)
    idx = np.random.permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx]


def main():
    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()

    print("Preprocessing: normalize [0,1]...")
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    X_tr, y_tr, X_val, y_val = split_validation(X_train, y_train)
    print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print("Fitting PCA (n_components={})...".format(N_PCA_COMPONENTS))
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_tr_pca = pca.fit_transform(X_tr)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    pca_mean = pca.mean_.copy()  # for deployment (optional; PCA.transform uses it internally)
    variance_retained = pca.explained_variance_ratio_.sum() * 100
    print(f"Variance retained: {variance_retained:.2f}%")

    print("Training Linear SVM (replicate MATLAB best)...")
    clf = LinearSVC(max_iter=5000, random_state=RANDOM_STATE, C=1.0)
    clf.fit(X_tr_pca, y_tr)

    for name, X, y in [("Val", X_val_pca, y_val), ("Test", X_test_pca, y_test)]:
        pred = clf.predict(X)
        acc = accuracy_score(y, pred)
        print(f"  {name} accuracy: {acc:.4f}")

    # Save for deployment
    joblib.dump({
        "classifier": clf,
        "pca": pca,
        "pca_mean": pca_mean,
        "n_components": N_PCA_COMPONENTS,
    }, os.path.join(OUT_DIR, "svm_pca_model.joblib"))
    print(f"Model saved to {OUT_DIR}/svm_pca_model.joblib")

    # Python deployment report (evidence)
    report_path = os.path.join(REPORT_DIR, "python_deployment_report.txt")
    with open(report_path, "w") as f:
        f.write("Python Deployment Training Report (replicates MATLAB pipeline)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Preprocessing: min-max [0,1]. Train/Val/Test split: 85/15 + holdout.\n")
        f.write(f"PCA: n_components={N_PCA_COMPONENTS}, variance retained={variance_retained:.2f}%\n")
        f.write(f"Model: LinearSVC. Test accuracy: {accuracy_score(y_test, clf.predict(X_test_pca)):.4f}\n")
        f.write("\nConfusion matrix (test):\n")
        f.write(str(confusion_matrix(y_test, clf.predict(X_test_pca))))
        f.write("\n")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
