## Fraud Detection

This project implements a machine learning-based fraud detection system using the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). The system uses logistic regression and random forest classifiers to identify fraudulent transactions. The models are optimized using grid search with cross-validation and evaluated using various performance metrics.

### Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Adjusting Model Sensitivity](#adjusting-model-sensitivity)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Confusion Matrix](#confusion-matrix)
- [License](#license)

### Features

- **Two models**: Logistic Regression and Random Forest classifiers.
- **Hyperparameter tuning**: Uses `GridSearchCV` to optimize model parameters.
- **Feature Engineering**: Adds new features based on transaction amounts and time-based features.
- **Model Evaluation**: Precision, recall, F1-score, and confusion matrix to assess model performance.
- **Class Imbalance Handling**: Ability to handle imbalanced data with techniques like `class_weight='balanced'` and SMOTE oversampling.
- **Interactive Model Selection**: Users can select the model (Logistic Regression or Random Forest) when running the script.

### Installation

#### Prerequisites

- Python 3.8+
- Libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `joblib`
- `imblearn` (for SMOTE, if using oversampling)
- `xgboost` (optional)

#### Setting up the environment

1. **Clone the repository**:

```bash
git clone git@github.com:fnyamweya/fraud-detection.git
cd fraud-detection
```

2. **Install dependencies**:
   It's recommended to use a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Download the dataset**:
   You can download the dataset from [Kaggle's Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), or if you have KaggleHub set up, use:

```python
import kagglehub
kagglehub.dataset_download("mlg-ulb/creditcardfraud")
```

4. **Place the dataset** in the `data/` folder:

```bash
mv path/to/downloaded/creditcard.csv data/
```

### Usage

#### Training and Evaluating Models

You can train and evaluate either the **Logistic Regression** or **Random Forest** model by running the script:

```bash
python main.py
```

Once the script starts, it will prompt you to choose which model to train:

```bash
Choose a model to train:
1. Random Forest
2. Logistic Regression
Enter 1 or 2:
```

After training, the model will output evaluation metrics, including accuracy, precision, recall, F1-score, and confusion matrix.

#### Adjusting Model Sensitivity

To adjust the model's sensitivity to fraud detection (increase recall and detect more fraud cases), you can manually adjust the decision threshold. By default, logistic regression uses a 0.5 threshold. You can adjust this to make the model more sensitive.

```python
# Example of evaluating the model with a threshold of 0.4
evaluate_model_with_threshold(model, X_test, y_test, threshold=0.4)
```

#### Handling Class Imbalance

1. **Class Weighting**: You can train the models with `class_weight='balanced'` to handle class imbalance automatically.
2. **Oversampling with SMOTE**:

To resample the training data using SMOTE, use the following code before training the model:

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

#### Confusion Matrix

The confusion matrix will be plotted after the model evaluation. It shows the true positive, false positive, true negative, and false negative counts, helping you visualize the model’s performance on fraud detection.

For example:

- **True Negatives**: Correctly predicted non-fraud cases.
- **False Positives**: Incorrectly predicted fraud cases when they are actually non-fraud.
- **False Negatives**: Missed fraud cases.
- **True Positives**: Correctly predicted fraud cases.

#### Models

- **Logistic Regression**: A linear model that is simple but effective for fraud detection when well-tuned.
- **Random Forest**: A non-linear ensemble model that typically offers more flexibility and better performance for detecting complex patterns.

Both models are saved in the `models/` directory after training:

- `models/logistic_regression_model.pkl`
- `models/random_forest_model.pkl`

#### Evaluation Metrics

The models are evaluated using the following metrics:

- **Precision**: Measures how many of the predicted frauds are actually fraud.
- **Recall**: Measures how many of the actual frauds were detected by the model.
- **F1-score**: Harmonic mean of precision and recall, providing a single measure of the model's accuracy.
- **Confusion Matrix**: Visualizes the performance by showing true positives, false positives, true negatives, and false negatives.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
