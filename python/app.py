"""
Streamlit app: upload a handwritten digit image to verify the classifier.
Uses the same preprocessing and PCA + SVM model as the MATLAB pipeline (replicated in Python).
Run from project root: streamlit run python/app.py
"""
import os
import numpy as np
import streamlit as st
from PIL import Image
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "python", "models", "svm_pca_model.joblib")
IMG_SIZE = 28


def load_model():
    if not os.path.isfile(MODEL_PATH):
        st.error(
            "Model not found. Run the Python training pipeline first: "
            "`python python/train_pipeline.py` (requires data in data/csv/)."
        )
        return None
    return joblib.load(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize to 28x28, grayscale, normalize to [0,1], flatten (match MATLAB pipeline)."""
    img = image.convert("L")  # grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float64)
    # If image was white-on-black, invert so digit is bright on dark (MNIST-like)
    if arr.mean() > 127:
        arr = 255 - arr
    arr = (arr - 0) / 255.0
    return arr.flatten()


def predict_digit(model_bundle, flattened: np.ndarray) -> int:
    """Apply PCA then classifier (same pipeline as MATLAB)."""
    pca = model_bundle["pca"]
    clf = model_bundle["classifier"]
    x = flattened.reshape(1, -1)
    x_pca = pca.transform(x)
    return int(clf.predict(x_pca)[0])


def main():
    st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")
    st.title("Handwritten Digit Recognition")
    st.markdown(
        "Upload an image of a single handwritten digit (0–9). "
        "The model uses the same preprocessing and classifier as the MATLAB pipeline (PCA + SVM)."
    )

    model_bundle = load_model()
    if model_bundle is None:
        return

    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded is None:
        st.info("Upload an image to get a prediction.")
        return

    image = Image.open(uploaded)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    flattened = preprocess_image(image)
    pred = predict_digit(model_bundle, flattened)

    with col2:
        st.metric("Predicted digit", pred)
        st.caption("Preprocessing: 28×28 grayscale, normalize [0,1], PCA (100 components), Linear SVM.")

    # Show small preview of preprocessed 28x28
    with st.expander("Preprocessed 28×28 input"):
        st.image(
            flattened.reshape(IMG_SIZE, IMG_SIZE),
            caption="Normalized 28×28 (as seen by model)",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
