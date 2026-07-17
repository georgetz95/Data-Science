from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "nhanes_adult_access_analysis.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    data = pd.read_csv(DATA_PATH)

    checks = [
        {
            "check": "rows_in_processed_file",
            "value": len(data),
            "status": "pass" if len(data) > 0 else "fail",
        },
        {
            "check": "unique_respondent_ids",
            "value": data["seqn"].nunique(),
            "status": "pass" if data["seqn"].is_unique else "fail",
        },
        {
            "check": "positive_interview_weights",
            "value": int((data["wtint2yr"] > 0).sum()),
            "status": "pass" if (data["wtint2yr"] > 0).all() else "fail",
        },
        {
            "check": "valid_binary_outcome_values",
            "value": sorted(data["fair_poor"].dropna().unique().tolist()),
            "status": "pass" if set(data["fair_poor"].dropna().unique()).issubset({0.0, 1.0}) else "fail",
        },
    ]
    pd.DataFrame(checks).to_csv(OUTPUT_DIR / "python_quality_checks.csv", index=False)

    selected = [
        "fair_poor",
        "insurance_status",
        "usual_care",
        "age_years",
        "gender",
        "race_ethnicity",
        "education",
        "poverty_ratio",
        "care_visits",
    ]
    model_data = data[selected].dropna(subset=["fair_poor", "insurance_status", "usual_care", "gender", "race_ethnicity", "education"])

    y = model_data["fair_poor"].astype(int)
    x = model_data.drop(columns=["fair_poor"])

    categorical = ["insurance_status", "usual_care", "gender", "race_ethnicity", "education", "care_visits"]
    numeric = ["age_years", "poverty_ratio"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=20260716,
        stratify=y,
    )

    model.fit(x_train, y_train)
    predicted_prob = model.predict_proba(x_test)[:, 1]
    predicted_class = (predicted_prob >= 0.50).astype(int)

    metrics = pd.DataFrame(
        [
            {"metric": "test_rows", "value": len(y_test)},
            {"metric": "event_rate", "value": float(np.mean(y_test))},
            {"metric": "roc_auc", "value": roc_auc_score(y_test, predicted_prob)},
            {"metric": "average_precision", "value": average_precision_score(y_test, predicted_prob)},
            {"metric": "accuracy_at_0_50", "value": accuracy_score(y_test, predicted_class)},
            {"metric": "precision_at_0_50", "value": precision_score(y_test, predicted_class, zero_division=0)},
            {"metric": "recall_at_0_50", "value": recall_score(y_test, predicted_class, zero_division=0)},
        ]
    )
    metrics["value"] = metrics["value"].round(4)
    metrics.to_csv(OUTPUT_DIR / "python_model_validation.csv", index=False)

    missing = (
        data[selected]
        .isna()
        .mean()
        .mul(100)
        .sort_values()
        .reset_index()
        .rename(columns={"index": "variable", 0: "missing_pct"})
    )
    missing.to_csv(OUTPUT_DIR / "python_missingness_summary.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.barh(missing["variable"], missing["missing_pct"], color="#5B8C85")
    plt.xlabel("Missing percent")
    plt.title("Python QA check: selected field missingness")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "python_missingness_check.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
