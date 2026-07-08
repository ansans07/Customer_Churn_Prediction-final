# Customer Churn Prediction 

This project predicts whether a telecom customer is likely to churn. It is designed as an interview-ready machine learning project using models covered in or closely related to the Machine Learning Specialization:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

No Gradient Boosting model from `sklearn.ensemble.GradientBoostingClassifier` is used in this version.

## Project Workflow

1. Load Telco Customer Churn dataset
2. Clean data
3. Perform EDA
4. Engineer useful features
5. Split data into train/test sets
6. Apply SMOTE only on the training data
7. Build preprocessing pipeline with OneHotEncoder and StandardScaler
8. Train four models using selected tuned hyperparameters
9. Evaluate models on unseen test data
10. Select the best model using F1 score
11. Save trained models and reports
12. Run a Streamlit web app for predictions

## Why F1 Score?

Customer churn prediction is a classification problem with class imbalance. Accuracy alone can be misleading. F1 score balances precision and recall, making it a better metric when both false positives and false negatives matter.

## Folder Structure

```text
Customer-Churn-EndToEnd/
├── app.py
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── figures/
├── models/
├── notebooks/
├── reports/
├── requirements.txt
└── src/
    ├── config.py
    ├── data_preprocessing.py
    ├── eda.py
    ├── evaluation.py
    ├── modeling.py
    ├── predict.py
    └── train.py
```

## Hyperparameter Tuning

The project includes an optional tuning script:

```bash
python src/tune_models.py
```

This performs cross-validation based hyperparameter search. It can take longer, so the main `train.py` uses final selected hyperparameters and runs quickly.

## How to Run

### 1. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train models

```bash
python src/train.py
```

This will train Logistic Regression, Decision Tree, Random Forest, and XGBoost, then create:

- trained models in `models/`
- metrics in `reports/`
- plots in `figures/`

### 4. Evaluate saved models

```bash
python src/evaluate.py
```

### 5. Run Streamlit app

```bash
streamlit run app.py
```

If Streamlit command does not work, use:

```bash
python -m streamlit run app.py
```

## Models Used

### Logistic Regression
Used as a strong baseline model. It predicts churn probability and is easy to interpret.

### Decision Tree
A tree-based model that makes decisions using feature splits. Easy to visualize and explain.

### Random Forest
An ensemble of many decision trees trained independently. It reduces overfitting compared with a single decision tree.

### XGBoost
An optimized boosting model often used for tabular machine learning problems. It is included because it is commonly taught after decision trees and random forests in many ML learning paths.

## Evaluation Metrics

The project reports:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Matthews Correlation Coefficient
- Confusion Matrix

## Important ML Concepts Used

- Train/test split
- Cross-validation
- Hyperparameter tuning script using cross-validation
- SMOTE for class imbalance
- One-hot encoding
- Feature scaling
- Model comparison
- Feature importance
- Probability prediction
- Business recommendations



