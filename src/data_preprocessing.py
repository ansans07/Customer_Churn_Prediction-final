import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from config import RANDOM_STATE, TEST_SIZE


def load_data(path):
    df = pd.read_csv(path)
    return df



def clean_data(df):
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    # TotalCharges contains blank strings in the Telco dataset
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    return df



def add_features(df):
    df = df.copy()
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["IsNewCustomer"] = (df["tenure"] <= 12).astype(int)
    df["IsLongTermCustomer"] = (df["tenure"] >= 36).astype(int)
    df["HighMonthlyCharges"] = (df["MonthlyCharges"] >= df["MonthlyCharges"].median()).astype(int)
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    ).astype(str)
    return df


def split_features_target(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def make_preprocessor(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def get_train_test_data(df):
    X, y = split_features_target(df)
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def get_feature_names(preprocessor):
    feature_names = []
    num_features = preprocessor.transformers_[0][2]
    feature_names.extend(num_features)
    cat_pipeline = preprocessor.named_transformers_["cat"]
    cat_encoder = cat_pipeline.named_steps["onehot"]
    cat_features = preprocessor.transformers_[1][2]
    feature_names.extend(cat_encoder.get_feature_names_out(cat_features).tolist())
    return feature_names
