from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from .config import RANDOM_STATE


def build_models(preprocessor):
    return {
        "Logistic Regression": ImbPipeline(steps=[
            ("preprocessor", clone(preprocessor)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": ImbPipeline(steps=[
            ("preprocessor", clone(preprocessor)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
        ]),
        "Random Forest": ImbPipeline(steps=[
            ("preprocessor", clone(preprocessor)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1))
        ]),
        "XGBoost": ImbPipeline(steps=[
            ("preprocessor", clone(preprocessor)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
                tree_method="hist",
            ))
        ]),
    }


def parameter_grids():
    # Small search spaces keep the project runnable on laptops while still showing hyperparameter tuning.
    return {
        "Logistic Regression": {
            "model__C": [0.1, 1],
            "model__solver": ["lbfgs"],
            "model__class_weight": [None, "balanced"],
        },
        "Decision Tree": {
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [3, 5],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
        "Random Forest": {
            "model__n_estimators": [50],
            "model__max_depth": [8, 12],
            "model__min_samples_leaf": [2, 4],
            "model__max_features": ["sqrt"],
        },
        "XGBoost": {
            "model__n_estimators": [50],
            "model__max_depth": [3, 4],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8],
            "model__colsample_bytree": [0.8],
        },
    }
