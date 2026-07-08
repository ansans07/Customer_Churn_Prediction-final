"""Optional hyperparameter tuning script.

Run this only when you want to perform cross-validation search. It takes longer than train.py.
The final project already contains a fast train.py using chosen tuned parameters.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
from sklearn.model_selection import RandomizedSearchCV
from src.config import DATA_PATH, CV_FOLDS, SCORING, RANDOM_STATE, REPORTS_DIR
from src.data_preprocessing import load_data, clean_data, add_features, get_train_test_data, make_preprocessor
from src.modeling import build_models, parameter_grids


def main():
    df = add_features(clean_data(load_data(DATA_PATH)))
    X_train, _, y_train, _ = get_train_test_data(df)
    preprocessor, _, _ = make_preprocessor(X_train)
    models = build_models(preprocessor)
    grids = parameter_grids()

    tuning_results = {}
    for name, pipeline in models.items():
        print(f"Tuning: {name}")
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=grids[name],
            n_iter=3,
            scoring=SCORING,
            cv=CV_FOLDS,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        tuning_results[name] = {
            "best_cv_score": float(search.best_score_),
            "best_params": search.best_params_,
        }

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "hyperparameter_tuning_results.json", "w") as f:
        json.dump(tuning_results, f, indent=4)
    print(json.dumps(tuning_results, indent=4))


if __name__ == "__main__":
    main()
