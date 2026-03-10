# Pipeline Documentation (Human-Friendly)

This folder explains each MATLAB script used in the MNIST digit classification assignment.

The goal is simple: make it easy for any team member, lecturer, or examiner to understand:
- what each script does
- why that step is needed
- what goes in and what comes out
- what formulas/ideas are being used

## Suggested Reading Order
1. `main_pipeline.md`
2. `load_data.md`
3. `preprocess_data.md`
4. `extract_features.md`
5. `train_models.md`
6. `evaluate_model.md`
7. `demo_gui.md`
8. `demo_walkthrough_gui.md`

## End-to-End Flow in One View
1. Load CSV data and split train/validation/test.
2. Normalize image pixels.
3. Build feature representations (Raw, PCA, HOG, Hybrid).
4. Train 3 classifier families on 5 feature sets.
5. Evaluate all 15 combinations with test metrics.
6. Use best model in GUI for final demo.

## Practical Tip
Always run MATLAB from project root so paths like `data/...` and `results/...` work correctly.

## Visual Evidence Included
- `preprocess_data.md` includes preprocessing snapshot.
- `evaluate_model.md` includes:
	- test accuracy comparison chart
	- per-digit performance chart
	- best model confusion matrix
- `demo_walkthrough_gui.md` includes a 3-step GUI walkthrough.

These images are stored as tracked files under `docs/assets/` so they are visible directly on GitHub.
