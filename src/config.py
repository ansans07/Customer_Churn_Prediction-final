from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = ROOT_DIR / "figures"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
SCORING = "f1"
