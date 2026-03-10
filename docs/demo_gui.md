# `demo_gui.m`

## What This Script Is
This is the presentation layer. It gives a simple user interface to test digit recognition using the best trained model.

## What User Can Do
- upload an image
- auto-convert to grayscale and 28x28
- run prediction
- see predicted digit and confidence-style output

## Internal Flow
1. Load best model info from evaluation results.
2. Load corresponding trained model.
3. On upload, preprocess image.
4. Extract matching feature type (Raw/PCA/HOG/Hybrid).
5. Predict digit and display output.

## Why It Is Useful
- Makes final demo much more engaging than command-line only.
- Shows practical application of trained model.
- Helps examiners quickly understand project impact.

## Notes
- GUI expects model artifacts to already exist.
- So run training and evaluation first.

## Good Demo Practice
- Test with clear handwritten digits first.
- Mention that confidence for some model types is approximate in current implementation.
- Explain that the backend model is selected from evaluation stage automatically.
