import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report, RocCurveDisplay,
    PrecisionRecallDisplay
)
from .config import FIGURES_DIR, REPORTS_DIR, MODELS_DIR


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "PR_AUC": average_precision_score(y_test, y_prob),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def save_classification_report(best_model, X_test, y_test):
    REPORTS_DIR.mkdir(exist_ok=True)
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    (REPORTS_DIR / "classification_report.txt").write_text(report)


def plot_confusion_matrix(best_model, X_test, y_test, model_name):
    FIGURES_DIR.mkdir(exist_ok=True)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=200)
    plt.close()


def plot_roc_pr_curves(models, X_test, y_test):
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    plt.title("ROC Curves")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    for name, model in models.items():
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    plt.title("Precision-Recall Curves")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curves.png", dpi=200)
    plt.close()


def plot_model_comparison(results_df):
    FIGURES_DIR.mkdir(exist_ok=True)
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    plot_df = results_df.set_index("Model")[metrics]
    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=200)
    plt.close()


def save_feature_importance(best_model, model_name, top_n=20):
    FIGURES_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    pipeline = best_model
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    try:
        from .data_preprocessing import get_feature_names
        feature_names = get_feature_names(preprocessor)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return
        fi = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi = fi.sort_values("importance", ascending=False).head(top_n)
        fi.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
        plt.figure(figsize=(8, 7))
        sns.barplot(data=fi, x="importance", y="feature")
        plt.title(f"Top {top_n} Feature Importance - {model_name}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=200)
        plt.close()
    except Exception as exc:
        (REPORTS_DIR / "feature_importance_error.txt").write_text(str(exc))


def save_artifacts(best_model, best_model_name, tuned_models, results_df, best_params):
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    for name, model in tuned_models.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{safe_name}.pkl")
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    (REPORTS_DIR / "best_model.txt").write_text(best_model_name)
    with open(REPORTS_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
