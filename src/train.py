import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import DATA_PATH
from src.data_preprocessing import load_data, clean_data, add_features, get_train_test_data, make_preprocessor
from src.modeling import build_models
from src.evaluation import (
    evaluate_model, save_artifacts, plot_confusion_matrix,
    plot_roc_pr_curves, plot_model_comparison, save_classification_report,
    save_feature_importance
)
from src.eda import run_eda


def main():
    print("Loading data...")
    raw_df = load_data(DATA_PATH)
    df = add_features(clean_data(raw_df))
    run_eda(df)

    X_train, X_test, y_train, y_test = get_train_test_data(df)
    preprocessor, _, _ = make_preprocessor(X_train)
    models = build_models(preprocessor)

    # Final tuned hyperparameters. These are intentionally lightweight so the project runs quickly on laptops.
    tuned_params = {
        "Logistic Regression": {
            "model__C": 1,
            "model__solver": "liblinear",
            "model__class_weight": "balanced",
        },
        "Decision Tree": {
            "model__criterion": "gini",
            "model__max_depth": 5,
            "model__min_samples_leaf": 2,
        },
        "Random Forest": {
            "model__n_estimators": 50,
            "model__max_depth": 8,
            "model__min_samples_leaf": 4,
            "model__max_features": "sqrt",
        },
        "XGBoost": {
            "model__n_estimators": 50,
            "model__max_depth": 3,
            "model__learning_rate": 0.1,
            "model__subsample": 0.8,
            "model__colsample_bytree": 0.8,
        },
    }

    fitted_models = {}
    results = []

    for name, model in models.items():
        print(f"Training: {name}", flush=True)
        model.set_params(**tuned_params[name])
        model.fit(X_train, y_train)
        fitted_models[name] = model
        results.append(evaluate_model(name, model, X_test, y_test))

    results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    save_artifacts(best_model, best_model_name, fitted_models, results_df, tuned_params)
    save_classification_report(best_model, X_test, y_test)
    plot_confusion_matrix(best_model, X_test, y_test, best_model_name)
    plot_roc_pr_curves(fitted_models, X_test, y_test)
    plot_model_comparison(results_df)
    save_feature_importance(best_model, best_model_name)

    print("\nTraining complete")
    print(f"Best model: {best_model_name}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
