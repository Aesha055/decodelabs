# Project 2: Data Classification Using AI

**DecodeLabs — AI Engineer Internship Track**

## Overview

This project implements a basic **supervised learning classification pipeline** from scratch logic using scikit-learn. It demonstrates how a machine learning model can learn patterns from labeled data and use them to classify new, unseen examples.

The project uses the classic **Iris flower dataset** — 150 samples of iris flowers, each described by 4 measurements, belonging to one of 3 species.

## What This Project Covers

| Step | Description |
|------|-------------|
| 1. Load & understand the dataset | Inspect shape, features, class balance, and summary statistics |
| 2. Train/test split | 80% training, 20% testing, stratified split with a fixed random seed |
| 3. Apply classification algorithms | Trains and compares **K-Nearest Neighbors** and **Decision Tree** classifiers |
| 4. Evaluate the model | Accuracy, precision/recall/F1 score, and a confusion matrix |
| 5. Predict on new data | Tests the best model against 3 brand-new, unseen flower samples |

## Dataset

- **Source:** `sklearn.datasets.load_iris()` (built-in, no download required)
- **Samples:** 150 (50 per class)
- **Features:** sepal length, sepal width, petal length, petal width (all in cm)
- **Classes:** `setosa`, `versicolor`, `virginica`

## Requirements

```
pandas
scikit-learn
```

Install with:

```bash
pip install pandas scikit-learn
```

## How to Run

### Option A — Locally
```bash
python3 project2_classification.py
```

### Option B — Google Colab
1. Open a new Colab notebook.
2. Paste in the script (no file upload needed — the Iris dataset is built into scikit-learn).
3. Run all cells.

> Note: Unlike a custom CSV dataset, this script does **not** require uploading any files — the Iris dataset loads directly from scikit-learn.

## Sample Output

```
Best model: K-Nearest Neighbors (k=3) with accuracy 100.00%

Confusion Matrix (rows = actual, cols = predicted):
                   pred_setosa  pred_versicolor  pred_virginica
actual_setosa               10                0               0
actual_versicolor            0               10               0
actual_virginica             0                0              10
```

## Results Summary

| Model | Accuracy |
|-------|----------|
| K-Nearest Neighbors (k=3) | 100.00% |
| Decision Tree | 93.33% |

KNN performed best on this run, achieving perfect classification on the held-out test set. The Decision Tree slightly confused `versicolor` and `virginica`, which are the two species with the most overlapping measurements.

## Project Structure

```
.
├── project2_classification.py   # Main script — full ML pipeline
└── README.md                    # This file
```

## Key Concepts Demonstrated

- **Supervised learning** — training a model on labeled data
- **Train/test split** — evaluating generalization on unseen data
- **Classification algorithms** — KNN (instance-based) vs. Decision Tree (rule-based)
- **Model evaluation** — accuracy, precision, recall, F1-score, confusion matrix
- **Inference** — using a trained model to predict on brand-new data

## Next Steps / Extensions

- Swap in a custom dataset (e.g. `lds1.csv`) instead of Iris
- Add data visualization (scatter plots, decision boundaries)
- Try additional algorithms (Logistic Regression, SVM, Random Forest)
- Add cross-validation instead of a single train/test split
- Tune hyperparameters (e.g. `k` in KNN, `max_depth` in Decision Tree)

## Author

DecodeLabs AI Engineering Internship — Project 2 Milestone
