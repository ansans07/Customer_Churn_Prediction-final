import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from src.config import DATA_PATH, MODELS_DIR, REPORTS_DIR
from src.data_preprocessing import load_data, clean_data, add_features, get_train_test_data
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_pr_curves, plot_model_comparison


def main():
    df = add_features(clean_data(load_data(DATA_PATH)))
    _, X_test, _, y_test = get_train_test_data(df)

    model_files = {
        "Logistic Regression": MODELS_DIR / "logistic_regression.pkl",
        "Decision Tree": MODELS_DIR / "decision_tree.pkl",
        "Random Forest": MODELS_DIR / "random_forest.pkl",
        "XGBoost": MODELS_DIR / "xgboost.pkl",
    }
    models = {name: joblib.load(path) for name, path in model_files.items() if path.exists()}
    if not models:
        raise FileNotFoundError("No trained models found. Run: python src/train.py")

    results = [evaluate_model(name, model, X_test, y_test) for name, model in models.items()]
    results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False)
    REPORTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(REPORTS_DIR / "evaluation_results.csv", index=False)
    best_name = results_df.iloc[0]["Model"]
    plot_confusion_matrix(models[best_name], X_test, y_test, best_name)
    plot_roc_pr_curves(models, X_test, y_test)
    plot_model_comparison(results_df)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
