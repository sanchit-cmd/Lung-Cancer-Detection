# Lung Cancer Detection

This project explores a lung cancer dataset with Python and scikit-learn. The
analysis is implemented in a Jupyter notebook and includes data cleaning,
visualization, feature preparation, and two classification models.

> **Important:** The notebook predicts the dataset's `survived` column. It is
an educational analysis, not a medical diagnostic tool, and its results should
not be used to guide healthcare decisions.

## Project contents

- [`Lung_cancer_Detection.ipynb`](./Lung_cancer_Detection.ipynb) — data analysis
  and model training workflow.
- [`pyproject.toml`](./pyproject.toml) — project metadata and dependencies.
- [`uv.lock`](./uv.lock) — locked dependency versions.

## Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)

## Setup

In the project directory, install the declared dependencies:

```bash
uv sync
```

The notebook uses KaggleHub to download the dataset
[`khwaishsaxena/lung-cancer-dataset`](https://www.kaggle.com/datasets/khwaishsaxena/lung-cancer-dataset)
when its first cell runs. An internet connection is required. Kaggle may
require account credentials or acceptance of the dataset's terms, depending
on its access settings.

## Run the notebook

Launch Jupyter Notebook with the project environment:

```bash
uv run --with notebook jupyter notebook Lung_cancer_Detection.ipynb
```

Alternatively, open `Lung_cancer_Detection.ipynb` in VS Code and select the
Python environment created by `uv sync` as the notebook kernel. Run the cells
from top to bottom.

## Analysis workflow

The notebook:

1. Downloads and loads the CSV dataset with KaggleHub and pandas.
2. Removes the `id` column and converts `age` to an integer.
3. Explores numeric feature distributions and correlations.
4. Creates a `treatment_duration` feature from the diagnosis and end-of-
   treatment dates, then removes those source date columns.
5. Encodes categorical columns and separates features from the `survived`
   target.
6. Splits the data into training and test sets, scales the features, and fits:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
7. Displays accuracy and confusion matrices for both models.

## Current limitations

- The notebook's distribution-plotting cell uses `np` but does not currently
  import NumPy as `np`. Add `import numpy as np` to the first cell before
  running that plotting cell.
- Preprocessing and scaling are performed before splitting the data. This can
  introduce data leakage, so the reported test metrics should be treated as
  exploratory rather than as an unbiased estimate of model performance.
- The notebook uses one train/test split and does not save a trained model or
  provide a prediction interface.
