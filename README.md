# Developer Role Classification Project

This repository contains two Jupyter notebooks for analyzing and modeling Git commit data to classify developer roles (e.g., backend, frontend, fullstack, QA). The project involves Exploratory Data Analysis (EDA), feature engineering, and machine learning (ML) modeling.

- **EDA AND FEATURE ENGINEERING (3).ipynb**: Focuses on data cleaning, exploration, and feature preparation.
- **ML MODEL (6).ipynb**: Builds and evaluates ML models for classification.

**Note**: There is no Deep Learning (DL) component in the provided notebooks. The sections below are structured as one for EDA/Feature Engineering and one for ML Modeling, assuming "DL" may have been a typo or misreference. If a DL model exists separately, update this README accordingly.

## Prerequisites

To reproduce the results, ensure the following:

- Python 3.8+ installed.
- The `final_dataset.csv` file (used in both notebooks) placed in the same directory as the notebooks.
- Install required dependencies via pip:

  ```bash
  pip install pandas numpy seaborn matplotlib scikit-learn nltk
  ```

- Download NLTK resources (run in Python console or add to notebook):

  ```python
  import nltk
  nltk.download('stopwords')
  nltk.download('punkt')  # If needed for tokenization
  ```

- Jupyter Notebook or JupyterLab installed:

  ```bash
  pip install jupyter
  ```

- Run the notebooks in order: Start with EDA/Feature Engineering, as it prepares insights used in the ML notebook.

## Section 1: Reproducing EDA and Feature Engineering

This section covers the "EDA AND FEATURE ENGINEERING (3).ipynb" notebook, which loads the dataset, performs data cleaning, handles unstructured data (e.g., file extensions, commit times), drops unnecessary columns, and generates visualizations for insights.

### Steps to Reproduce

1. **Set Up Environment**:
   - Open a terminal and navigate to the directory containing the notebook and `final_dataset.csv`.
   - Launch Jupyter:

     ```bash
     jupyter notebook
     ```

2. **Run the Notebook**:
   - Open "EDA AND FEATURE ENGINEERING (3).ipynb".
   - Execute cells sequentially (Shift + Enter).
   - Key steps include:
     - Loading the dataset: `df = pd.read_csv('final_dataset.csv')`.
     - Extracting unique file extensions and categorizing commit times (e.g., into "Morning", "Afternoon", etc.).
     - Generating visualizations:
       - Count plots for `committype` vs. `role`.
       - Bar plots for numerical features (e.g., `numfileschanged`, `linesadded`) vs. `role`.
       - Time-based count plots.
   - No outputs are saved; visualizations will display inline in the notebook.

3. **Expected Outputs**:
   - Cleaned DataFrame with new columns like `day` and `time`.
   - Printed unique file extensions (e.g., `['js_ts', 'css', ...]`).
   - Seaborn plots showing distributions and insights (e.g., QA developers work more in evenings/nights).
   - No saved files; results are exploratory and inform the next notebook.

4. **Troubleshooting**:
   - If plots don't display, ensure `matplotlib` inline mode: `%matplotlib inline`.
   - Missing data file: Ensure `final_dataset.csv` exists with columns like `role`, `committype`, `fileextensions`, etc.
   - Runtime: <1 minute on standard hardware.

## Section 2: Reproducing ML Model

This section covers the "ML MODEL (6).ipynb" notebook, which performs advanced feature engineering (e.g., one-hot encoding, TF-IDF on commit messages), trains models (Logistic Regression, Decision Tree, Random Forest), tunes hyperparameters, and evaluates performance.

### Steps to Reproduce

1. **Set Up Environment**:
   - Same as above: Navigate to the directory and launch Jupyter if not already running.

2. **Run the Notebook**:
   - Open "ML MODEL (6).ipynb".
   - Execute cells sequentially.
   - Key steps include:
     - Loading and preprocessing: Similar to EDA (file extensions, time categorization).
     - Text processing: Clean commit messages with lowercase, punctuation removal, stopword removal, and stemming.
     - Feature encoding: MultiLabelBinarizer for file extensions, TF-IDF for commit messages, one-hot for categoricals.
     - Train-test split: 80/20 split.
     - Models:
       - Baseline: Logistic Regression (accuracy ~0.94).
       - Decision Tree: Initial (~0.94), tuned with GridSearchCV (~0.967).
       - Random Forest: Best performer (~0.973 accuracy, 0.969 macro F1).
     - Evaluation: Accuracy, macro F1, confusion matrix, classification report.
   - Hyperparameter tuning uses GridSearchCV; it may take a few minutes.

3. **Expected Outputs**:
   - Printed metrics:
     - Logistic Regression: Accuracy 0.94, Macro F1 0.9305.
     - Decision Tree (tuned): Accuracy 0.9667, Macro F1 0.9635.
     - Random Forest: Accuracy 0.9733, Macro F1 0.9694.
   - Visualizations: Confusion matrix heatmap for Random Forest.
   - Classification report detailing precision/recall/F1 per role (e.g., backend: 1.00 F1).
   - No saved models/files; results are printed inline.

4. **Troubleshooting**:
   - NLTK errors: Ensure stopwords are downloaded.
   - Long runtime: GridSearchCV for Decision Tree; reduce param grid if needed (e.g., fewer max_depth values).
   - Reproducibility: Set random_state=42 in train_test_split and models for consistent splits.
   - Runtime: ~5-10 minutes (due to tuning and TF-IDF).

## Additional Notes

- **Dataset**: Assumes `final_dataset.csv` has 1500 entries with no missing values. If regenerating data, ensure consistency.
- **Reproducibility**: Results may vary slightly due to random splits; use fixed seeds.
- **Extensions**: For DL (e.g., neural networks), add a new notebook with libraries like TensorFlow/Keras, using embeddings for text features.
- **Contact**: For issues, check console errors or provide dataset sample.

This README ensures full reproducibility. Update as needed for new features!
