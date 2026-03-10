# `extract_features.m`

## What This Script Is
This is the feature engineering stage. It transforms image data into multiple representations and stores all of them for model comparison.

## Feature Sets Built
1. Raw pixels (baseline)
2. PCA-50
3. PCA-100
4. HOG
5. Hybrid (HOG + PCA)

## Why Multiple Features
Different models respond differently to representation.
This stage gives evidence for comparative analysis in assignment report.

## Core Ideas (Layman Version)
### Raw Pixels
- Directly use 784 pixel values.
- Simple baseline.

### PCA (Principal Component Analysis)
- Compresses data into fewer directions carrying most variance.
- Reduces noise and computation.

Formula intuition:
- Project data to top components:
`Z = X * W`
where `W` contains selected principal directions.

Variance explained:
`explained = sum(top eigenvalues) / sum(all eigenvalues)`

### HOG (Histogram of Oriented Gradients)
- Focuses on edges and stroke directions.
- Very suitable for digit shapes.

### Hybrid (HOG + PCA)
- First get shape-rich HOG features.
- Then compress with PCA for efficiency.
- Keeps strong shape signal while reducing dimension.

## Output
Saved to:
- `data/feature-extracted/features.mat`

Includes all train/val/test feature matrices and PCA coefficients.

## Why This Helps Assignment Marks
- Clear feature comparison
- Includes baseline + advanced method
- Provides a defendable custom approach (Hybrid)
