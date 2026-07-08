import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .config import FIGURES_DIR, REPORTS_DIR


def run_eda(df):
    FIGURES_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    overview = []
    overview.append(f"Rows: {df.shape[0]}")
    overview.append(f"Columns: {df.shape[1]}")
    overview.append("\nMissing values:\n" + df.isna().sum().to_string())
    overview.append("\nDuplicate rows: " + str(df.duplicated().sum()))
    overview.append("\nChurn distribution:\n" + df["Churn"].value_counts(normalize=True).to_string())
    (REPORTS_DIR / "eda_summary.txt").write_text("\n".join(overview))

    plt.figure(figsize=(5, 4))
    df["Churn"].value_counts().sort_index().plot(kind="bar")
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_distribution.png", dpi=200)
    plt.close()

    for col in ["Contract", "InternetService", "PaymentMethod"]:
        plt.figure(figsize=(8, 4))
        pd.crosstab(df[col], df["Churn"]).plot(kind="bar", ax=plt.gca())
        plt.title(f"Churn by {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=25, ha="right")
        plt.legend(title="Churn")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"churn_by_{col}.png", dpi=200)
        plt.close()

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x="Churn", y=col)
        plt.title(f"{col} vs Churn")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{col}_vs_churn.png", dpi=200)
        plt.close()

    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include="number")
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()
