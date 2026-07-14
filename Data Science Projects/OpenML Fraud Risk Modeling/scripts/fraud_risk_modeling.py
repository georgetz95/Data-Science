from pathlib import Path
import json
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data" / "raw" / "creditcard_openml_1597.parquet"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIGURE_DIR = PROJECT_DIR / "figures"
TABLE_DIR = PROJECT_DIR / "outputs" / "tables"
MODEL_DIR = PROJECT_DIR / "outputs" / "model_artifacts"

for folder in [PROCESSED_DIR, FIGURE_DIR, TABLE_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def pct(x, digits=2):
    return round(100 * x, digits)


def top_review_metrics(y_true, scores, review_rate):
    scored = pd.DataFrame({"actual": y_true, "score": scores}).sort_values("score", ascending=False)
    n_review = max(1, int(np.ceil(len(scored) * review_rate)))
    reviewed = scored.head(n_review)
    fraud_total = scored["actual"].sum()
    fraud_found = reviewed["actual"].sum()

    return {
        "review_rate": review_rate,
        "reviewed_transactions": n_review,
        "fraud_found": int(fraud_found),
        "precision_at_review_rate": fraud_found / n_review,
        "fraud_capture_rate": fraud_found / fraud_total if fraud_total > 0 else np.nan,
        "lift_over_random": (fraud_found / n_review) / scored["actual"].mean(),
    }


def population_stability_index(expected, actual, bins=10):
    expected = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna()

    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))

    if len(cuts) < 3:
        cuts = np.linspace(expected.min(), expected.max(), bins + 1)

    expected_counts = pd.cut(expected, bins=cuts, include_lowest=True, duplicates="drop").value_counts(sort=False)
    actual_counts = pd.cut(actual, bins=cuts, include_lowest=True, duplicates="drop").value_counts(sort=False)

    expected_share = expected_counts / expected_counts.sum()
    actual_share = actual_counts / actual_counts.sum()

    expected_share = expected_share.replace(0, 0.0001)
    actual_share = actual_share.replace(0, 0.0001)

    return float(((actual_share - expected_share) * np.log(actual_share / expected_share)).sum())


def make_temporal_split(frame):
    ordered = frame.sort_values("Time").reset_index(drop=True)
    n = len(ordered)

    ordered["split"] = "train"
    ordered.loc[int(n * 0.70):int(n * 0.85) - 1, "split"] = "validation"
    ordered.loc[int(n * 0.85):, "split"] = "test"

    return ordered


def engineer_features(frame):
    out = frame.copy()
    out["amount_log"] = np.log1p(out["Amount"])
    out["time_hours"] = out["Time"] / 3600
    out["hour_of_day"] = (out["time_hours"] % 24).astype(float)
    out["amount_is_zero"] = (out["Amount"] == 0).astype(int)
    return out


def fit_models(train, validation, feature_cols):
    X_train = train[feature_cols]
    y_train = train["Class"].astype(int)
    X_val = validation[feature_cols]
    y_val = validation["Class"].astype(int)

    models = {
        "balanced_logistic_regression": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1200,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "balanced_random_forest": RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }

    rows = []
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_val)[:, 1]
        rows.append(
            {
                "model": name,
                "split": "validation",
                "roc_auc": roc_auc_score(y_val, scores),
                "average_precision": average_precision_score(y_val, scores),
                "brier_score": brier_score_loss(y_val, scores),
            }
        )
        fitted[name] = model

    model_table = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    best_model_name = model_table.iloc[0]["model"]

    return fitted, model_table, best_model_name


def evaluate_model(model, model_name, split_name, frame, feature_cols):
    y = frame["Class"].astype(int)
    scores = model.predict_proba(frame[feature_cols])[:, 1]

    base = {
        "model": model_name,
        "split": split_name,
        "transactions": len(frame),
        "fraud_cases": int(y.sum()),
        "fraud_rate": y.mean(),
        "roc_auc": roc_auc_score(y, scores),
        "average_precision": average_precision_score(y, scores),
        "brier_score": brier_score_loss(y, scores),
    }

    top_rows = []
    for rate in [0.005, 0.01, 0.02, 0.05]:
        row = top_review_metrics(y, scores, rate)
        row.update({"model": model_name, "split": split_name})
        top_rows.append(row)

    return base, pd.DataFrame(top_rows), scores


def plot_class_balance(split_summary):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = split_summary["split"]
    fraud_rates = split_summary["fraud_rate"] * 100
    ax.bar(labels, fraud_rates, color="#2B6CB0")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud prevalence is low across temporal splits")
    for i, value in enumerate(fraud_rates):
        ax.text(i, value + 0.01, f"{value:.3f}%", ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fraud_rate_by_split.png", dpi=300)
    plt.close(fig)


def plot_precision_recall(y_true, scores, average_precision):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(recall, precision, color="#2B6CB0", linewidth=2)
    ax.axhline(y_true.mean(), color="#718096", linestyle="--", linewidth=1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve on temporal test window")
    ax.text(0.05, 0.88, f"Average precision = {average_precision:.3f}", transform=ax.transAxes)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "precision_recall_curve.png", dpi=300)
    plt.close(fig)


def plot_review_budget(review_table):
    plot_data = review_table.copy()
    plot_data["review_rate_pct"] = plot_data["review_rate"] * 100
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(plot_data["review_rate_pct"], plot_data["fraud_capture_rate"] * 100, marker="o", color="#B83280")
    ax.set_xlabel("Manual review budget (% of transactions)")
    ax.set_ylabel("Fraud captured (%)")
    ax.set_title("Fraud capture rises quickly in the highest-risk score band")
    for _, row in plot_data.iterrows():
        ax.text(row["review_rate_pct"], row["fraud_capture_rate"] * 100 + 1, f"{row['fraud_capture_rate']*100:.1f}%", ha="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "review_budget_capture.png", dpi=300)
    plt.close(fig)


def plot_score_distribution(scored_test):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    good = scored_test.loc[scored_test["Class"] == 0, "fraud_score"]
    fraud = scored_test.loc[scored_test["Class"] == 1, "fraud_score"]
    ax.hist(good, bins=50, alpha=0.7, label="Non-fraud", density=True, color="#718096")
    ax.hist(fraud, bins=50, alpha=0.7, label="Fraud", density=True, color="#C48A1A")
    ax.set_xlabel("Model fraud score")
    ax.set_ylabel("Density")
    ax.set_title("Fraud cases concentrate in the high-score tail")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "score_distribution.png", dpi=300)
    plt.close(fig)


def plot_monitoring(monitoring):
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(monitoring["time_bucket"], monitoring["avg_score"], marker="o", color="#2B6CB0", label="Average score")
    ax1.set_ylabel("Average score")
    ax1.set_xlabel("Test-window time bucket")
    ax2 = ax1.twinx()
    ax2.bar(monitoring["time_bucket"], monitoring["alert_rate"] * 100, alpha=0.25, color="#C48A1A", label="Alert rate")
    ax2.set_ylabel("Alert rate (%)")
    ax1.set_title("Monitoring view: score level and alert volume over time")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "monitoring_by_time.png", dpi=300)
    plt.close(fig)


def plot_psi(psi_table):
    plot_data = psi_table.sort_values("psi", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.barh(plot_data["feature"], plot_data["psi"], color="#2B6CB0")
    ax.axvline(0.10, color="#C48A1A", linestyle="--", linewidth=1)
    ax.axvline(0.25, color="#B83280", linestyle="--", linewidth=1)
    ax.set_xlabel("Population stability index")
    ax.set_title("Feature drift check between train and test windows")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_drift_psi.png", dpi=300)
    plt.close(fig)


def main():
    data = pd.read_parquet(RAW_FILE)
    data.columns = [str(c) for c in data.columns]
    data["Class"] = data["Class"].astype(int)

    feature_data = engineer_features(data)
    split_data = make_temporal_split(feature_data)

    feature_cols = [c for c in split_data.columns if c not in ["Class", "split"]]

    train = split_data.query("split == 'train'").copy()
    validation = split_data.query("split == 'validation'").copy()
    test = split_data.query("split == 'test'").copy()

    fitted, validation_table, best_model_name = fit_models(train, validation, feature_cols)
    best_model = fitted[best_model_name]

    all_metric_rows = validation_table.to_dict("records")
    review_rows = []
    score_frames = {}

    for split_name, split_frame in [("train", train), ("validation", validation), ("test", test)]:
        metric_row, review_table, scores = evaluate_model(best_model, best_model_name, split_name, split_frame, feature_cols)
        all_metric_rows.append(metric_row)
        review_rows.append(review_table)
        score_frames[split_name] = split_frame.assign(fraud_score=scores)

    model_metrics = pd.DataFrame(all_metric_rows).drop_duplicates(subset=["model", "split"], keep="last")
    review_metrics = pd.concat(review_rows, ignore_index=True)

    validation_scores = score_frames["validation"]["fraud_score"]
    review_rate = 0.01
    threshold = float(np.quantile(validation_scores, 1 - review_rate))

    scored_test = score_frames["test"].copy()
    scored_test["alert"] = (scored_test["fraud_score"] >= threshold).astype(int)
    cm = confusion_matrix(scored_test["Class"], scored_test["alert"], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    threshold_summary = pd.DataFrame(
        [
            {
                "selected_model": best_model_name,
                "review_rate_target": review_rate,
                "validation_threshold": threshold,
                "test_alerts": int(scored_test["alert"].sum()),
                "test_transactions": len(scored_test),
                "test_alert_rate": scored_test["alert"].mean(),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "precision": tp / (tp + fp) if (tp + fp) else np.nan,
                "recall": tp / (tp + fn) if (tp + fn) else np.nan,
                "fraud_rate_in_alerts": scored_test.loc[scored_test["alert"] == 1, "Class"].mean(),
            }
        ]
    )

    split_summary = split_data.groupby("split", sort=False).agg(
        transactions=("Class", "size"),
        fraud_cases=("Class", "sum"),
        fraud_rate=("Class", "mean"),
        min_time=("Time", "min"),
        max_time=("Time", "max"),
        median_amount=("Amount", "median"),
        mean_amount=("Amount", "mean"),
    ).reset_index()

    monitoring = scored_test.copy()
    monitoring["time_bucket"] = pd.qcut(monitoring["Time"], q=8, labels=False, duplicates="drop") + 1
    monitoring = monitoring.groupby("time_bucket").agg(
        transactions=("Class", "size"),
        fraud_cases=("Class", "sum"),
        fraud_rate=("Class", "mean"),
        avg_score=("fraud_score", "mean"),
        alert_rate=("alert", "mean"),
        alerts=("alert", "sum"),
    ).reset_index()

    drift_features = ["Amount", "amount_log", "hour_of_day"] + [f"V{i}" for i in range(1, 9)]
    psi_table = pd.DataFrame(
        {
            "feature": feature,
            "psi": population_stability_index(train[feature], test[feature]),
        }
        for feature in drift_features
    ).sort_values("psi", ascending=False)

    rf_model = fitted["balanced_random_forest"]
    rf_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    logistic_model = fitted["balanced_logistic_regression"]
    logistic_coefficients = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": logistic_model.named_steps["model"].coef_[0],
            "abs_coefficient": np.abs(logistic_model.named_steps["model"].coef_[0]),
        }
    ).sort_values("abs_coefficient", ascending=False)

    data_quality = pd.DataFrame(
        [
            {"check": "rows", "value": len(split_data)},
            {"check": "columns", "value": split_data.shape[1]},
            {"check": "fraud_cases", "value": int(split_data["Class"].sum())},
            {"check": "fraud_rate", "value": split_data["Class"].mean()},
            {"check": "missing_cells", "value": int(split_data.isna().sum().sum())},
            {"check": "duplicate_rows", "value": int(split_data.duplicated().sum())},
            {"check": "selected_model", "value": best_model_name},
        ]
    )

    plot_class_balance(split_summary)
    test_ap = model_metrics.query("split == 'test'")["average_precision"].iloc[0]
    plot_precision_recall(scored_test["Class"], scored_test["fraud_score"], test_ap)
    plot_review_budget(review_metrics.query("split == 'test'"))
    plot_score_distribution(scored_test)
    plot_monitoring(monitoring)
    plot_psi(psi_table.head(10))

    split_data.to_csv(PROCESSED_DIR / "creditcard_feature_table.csv", index=False)
    scored_test.to_csv(PROCESSED_DIR / "test_scoring_output.csv", index=False)

    split_summary.to_csv(TABLE_DIR / "split_summary.csv", index=False)
    model_metrics.to_csv(TABLE_DIR / "model_metrics.csv", index=False)
    review_metrics.to_csv(TABLE_DIR / "review_budget_metrics.csv", index=False)
    threshold_summary.to_csv(TABLE_DIR / "threshold_summary.csv", index=False)
    monitoring.to_csv(TABLE_DIR / "monitoring_by_time_bucket.csv", index=False)
    psi_table.to_csv(TABLE_DIR / "feature_drift_psi.csv", index=False)
    rf_importance.to_csv(TABLE_DIR / "random_forest_feature_importance.csv", index=False)
    logistic_coefficients.to_csv(TABLE_DIR / "logistic_coefficients.csv", index=False)
    data_quality.to_csv(TABLE_DIR / "data_quality_checks.csv", index=False)

    db_path = PROJECT_DIR / "outputs" / "fraud_risk_scoring.sqlite"
    with sqlite3.connect(db_path) as conn:
        split_data.to_sql("fraud_feature_table", conn, if_exists="replace", index=False)
        scored_test.to_sql("fraud_test_scores", conn, if_exists="replace", index=False)
        threshold_summary.to_sql("model_threshold_summary", conn, if_exists="replace", index=False)
        monitoring.to_sql("model_monitoring_by_bucket", conn, if_exists="replace", index=False)

    metadata = {
        "source": "OpenML creditcard dataset, data_id=1597",
        "rows": int(len(split_data)),
        "fraud_cases": int(split_data["Class"].sum()),
        "fraud_rate": float(split_data["Class"].mean()),
        "selected_model": best_model_name,
        "feature_count": len(feature_cols),
        "threshold_review_rate": review_rate,
        "threshold": threshold,
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary_lines = [
        "SentiLink-style fraud risk modeling project summary",
        f"Dataset: OpenML creditcard, {len(split_data):,} transactions and {int(split_data['Class'].sum()):,} fraud cases.",
        f"Fraud prevalence: {pct(split_data['Class'].mean(), 3)}%.",
        f"Selected model: {best_model_name}.",
        f"Test average precision: {test_ap:.4f}.",
        f"At a validation-tuned 1% review budget, test recall is {threshold_summary['recall'].iloc[0]:.3f} and precision is {threshold_summary['precision'].iloc[0]:.3f}.",
        f"SQLite scoring database: {db_path.name}.",
    ]
    (PROJECT_DIR / "outputs" / "analysis_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
