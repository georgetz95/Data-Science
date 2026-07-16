from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlencode

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = PROJECT_DIR / "figures"
SQL_DIR = PROJECT_DIR / "sql"

CMS_DATASETS = {
    2023: "0e9f2f2b-7bf9-451a-912c-e02e654dd725",
    2024: "335e5f35-eca6-482d-87b3-f99883e213e3",
}

STATES = ["IN", "IL"]

HCPCS_CODES = {
    "99203": "New outpatient E/M",
    "99204": "New outpatient E/M",
    "99213": "Established outpatient E/M",
    "99214": "Established outpatient E/M",
    "99215": "Established outpatient E/M",
    "99232": "Subsequent hospital care",
    "99285": "Emergency department E/M",
    "93000": "Electrocardiogram",
    "97110": "Therapeutic procedure",
}

PAGE_SIZE = 5000

COLUMN_MAP = {
    "Rndrng_NPI": "rendering_npi",
    "Rndrng_Prvdr_Last_Org_Name": "provider_last_or_org_name",
    "Rndrng_Prvdr_First_Name": "provider_first_name",
    "Rndrng_Prvdr_MI": "provider_middle_initial",
    "Rndrng_Prvdr_Crdntls": "provider_credentials",
    "Rndrng_Prvdr_Ent_Cd": "provider_entity_type",
    "Rndrng_Prvdr_City": "provider_city",
    "Rndrng_Prvdr_State_Abrvtn": "provider_state",
    "Rndrng_Prvdr_Zip5": "provider_zip5",
    "Rndrng_Prvdr_RUCA": "provider_ruca",
    "Rndrng_Prvdr_RUCA_Desc": "provider_ruca_description",
    "Rndrng_Prvdr_Cntry": "provider_country",
    "Rndrng_Prvdr_Type": "provider_type",
    "Rndrng_Prvdr_Mdcr_Prtcptg_Ind": "medicare_participating",
    "HCPCS_Cd": "hcpcs_code",
    "HCPCS_Desc": "hcpcs_description",
    "HCPCS_Drug_Ind": "hcpcs_drug_indicator",
    "Place_Of_Srvc": "place_of_service",
    "Tot_Benes": "total_beneficiaries",
    "Tot_Srvcs": "total_services",
    "Tot_Bene_Day_Srvcs": "total_beneficiary_day_services",
    "Avg_Sbmtd_Chrg": "average_submitted_charge",
    "Avg_Mdcr_Alowd_Amt": "average_medicare_allowed",
    "Avg_Mdcr_Pymt_Amt": "average_medicare_payment",
    "Avg_Mdcr_Stdzd_Amt": "average_medicare_standardized",
}

NUMERIC_COLUMNS = [
    "total_beneficiaries",
    "total_services",
    "total_beneficiary_day_services",
    "average_submitted_charge",
    "average_medicare_allowed",
    "average_medicare_payment",
    "average_medicare_standardized",
]


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, TABLE_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def cms_url(dataset_id: str, params: dict) -> str:
    return f"https://data.cms.gov/data-api/v1/dataset/{dataset_id}/data?{urlencode(params)}"


def fetch_cms_slice(year: int, state: str, hcpcs_code: str) -> pd.DataFrame:
    raw_file = RAW_DIR / f"cms_provider_service_{year}_{state}_{hcpcs_code}.csv"
    if raw_file.exists():
        return pd.read_csv(raw_file, dtype=str)

    rows = []
    offset = 0
    while True:
        params = {
            "size": PAGE_SIZE,
            "offset": offset,
            "filter[Rndrng_Prvdr_State_Abrvtn]": state,
            "filter[HCPCS_Cd]": hcpcs_code,
        }
        response = requests.get(cms_url(CMS_DATASETS[year], params), timeout=60)
        response.raise_for_status()
        payload = response.json()
        page = payload if isinstance(payload, list) else payload.get("value", [])
        if not page:
            break
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.15)

    frame = pd.DataFrame(rows)
    frame.to_csv(raw_file, index=False)
    return frame


def download_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    quality_rows = []

    for year in CMS_DATASETS:
        for state in STATES:
            for hcpcs_code in HCPCS_CODES:
                frame = fetch_cms_slice(year, state, hcpcs_code)
                quality_rows.append(
                    {
                        "year": year,
                        "state": state,
                        "hcpcs_code": hcpcs_code,
                        "rows_downloaded": len(frame),
                        "source_dataset_id": CMS_DATASETS[year],
                    }
                )
                if len(frame):
                    frame["year"] = year
                    frame["extract_state"] = state
                    frame["extract_hcpcs_code"] = hcpcs_code
                    frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(TABLE_DIR / "download_quality_summary.csv", index=False)
    return raw, quality


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    claims = raw.rename(columns=COLUMN_MAP).copy()
    keep_cols = list(COLUMN_MAP.values()) + ["year", "extract_state", "extract_hcpcs_code"]
    claims = claims[keep_cols]

    for col in NUMERIC_COLUMNS:
        claims[col] = pd.to_numeric(claims[col], errors="coerce")

    claims["provider_name"] = np.where(
        claims["provider_entity_type"].eq("O"),
        claims["provider_last_or_org_name"],
        (
            claims["provider_last_or_org_name"].fillna("")
            + ", "
            + claims["provider_first_name"].fillna("")
        ).str.strip(", "),
    )
    claims["code_family"] = claims["hcpcs_code"].map(HCPCS_CODES)
    claims["place_of_service_label"] = claims["place_of_service"].map({"F": "Facility", "O": "Office"}).fillna("Other")

    claims = claims.drop_duplicates(["year", "rendering_npi", "provider_state", "hcpcs_code", "place_of_service"])
    claims = claims.dropna(subset=["rendering_npi", "provider_state", "hcpcs_code", "total_services", "average_medicare_payment"])

    claims["estimated_submitted_charge"] = claims["total_services"] * claims["average_submitted_charge"]
    claims["estimated_allowed_amount"] = claims["total_services"] * claims["average_medicare_allowed"]
    claims["estimated_medicare_payment"] = claims["total_services"] * claims["average_medicare_payment"]
    claims["estimated_standardized_payment"] = claims["total_services"] * claims["average_medicare_standardized"]
    claims["services_per_beneficiary"] = claims["total_services"] / claims["total_beneficiaries"].replace(0, np.nan)
    claims["beneficiary_days_per_beneficiary"] = claims["total_beneficiary_day_services"] / claims["total_beneficiaries"].replace(0, np.nan)
    claims["payment_per_beneficiary"] = claims["estimated_medicare_payment"] / claims["total_beneficiaries"].replace(0, np.nan)
    claims["charge_to_allowed_ratio"] = claims["average_submitted_charge"] / claims["average_medicare_allowed"].replace(0, np.nan)
    claims["payment_to_allowed_ratio"] = claims["average_medicare_payment"] / claims["average_medicare_allowed"].replace(0, np.nan)

    claims = claims.replace([np.inf, -np.inf], np.nan)
    claims.to_csv(PROCESSED_DIR / "provider_service_claims_clean.csv", index=False)
    return claims


def write_sqlite(claims: pd.DataFrame) -> dict[str, pd.DataFrame]:
    db_path = PROCESSED_DIR / "claims_audit_targeting.sqlite"
    if db_path.exists():
        db_path.unlink()

    outputs = {}
    with sqlite3.connect(db_path) as conn:
        claims.to_sql("raw_claims", conn, index=False, if_exists="replace")
        with open(SQL_DIR / "claims_audit_views.sql", "r", encoding="utf-8") as sql_file:
            conn.executescript(sql_file.read())

        queries = {
            "yearly_state_summary": "SELECT * FROM yearly_state_summary ORDER BY year, provider_state",
            "provider_payment_profile": "SELECT * FROM provider_payment_profile ORDER BY estimated_medicare_payment DESC",
            "hcpcs_specialty_summary": "SELECT * FROM hcpcs_specialty_summary ORDER BY estimated_medicare_payment DESC",
            "high_volume_code_summary": "SELECT * FROM high_volume_code_summary ORDER BY estimated_medicare_payment DESC",
        }
        for name, query in queries.items():
            table = pd.read_sql_query(query, conn)
            table.to_csv(TABLE_DIR / f"{name}.csv", index=False)
            outputs[name] = table

    return outputs


def percentile_by_group(frame: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.Series:
    return frame.groupby(group_cols)[value_col].rank(pct=True, method="average")


def build_audit_scores(claims: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = claims.copy()
    benchmark_group = ["year", "provider_state", "hcpcs_code", "place_of_service"]

    scored["payment_percentile"] = percentile_by_group(scored, benchmark_group, "estimated_medicare_payment")
    scored["services_per_bene_percentile"] = percentile_by_group(scored, benchmark_group, "services_per_beneficiary")
    scored["charge_ratio_percentile"] = percentile_by_group(scored, benchmark_group, "charge_to_allowed_ratio")
    scored["payment_per_bene_percentile"] = percentile_by_group(scored, benchmark_group, "payment_per_beneficiary")
    scored["beneficiary_day_percentile"] = percentile_by_group(scored, benchmark_group, "beneficiary_days_per_beneficiary")

    provider_code = (
        scored.groupby(["year", "provider_state", "rendering_npi", "hcpcs_code"], as_index=False)
        .agg(
            provider_code_services=("total_services", "sum"),
            provider_code_payment=("estimated_medicare_payment", "sum"),
        )
        .sort_values(["provider_state", "rendering_npi", "hcpcs_code", "year"])
    )
    prior = provider_code.copy()
    prior["year"] = prior["year"] + 1
    prior = prior.rename(
        columns={
            "provider_code_services": "prior_provider_code_services",
            "provider_code_payment": "prior_provider_code_payment",
        }
    )
    provider_code = provider_code.merge(
        prior[["year", "provider_state", "rendering_npi", "hcpcs_code", "prior_provider_code_services", "prior_provider_code_payment"]],
        on=["year", "provider_state", "rendering_npi", "hcpcs_code"],
        how="left",
    )
    provider_code["service_growth_rate"] = (
        provider_code["provider_code_services"] - provider_code["prior_provider_code_services"]
    ) / provider_code["prior_provider_code_services"].replace(0, np.nan)
    provider_code["payment_growth_rate"] = (
        provider_code["provider_code_payment"] - provider_code["prior_provider_code_payment"]
    ) / provider_code["prior_provider_code_payment"].replace(0, np.nan)

    scored = scored.merge(
        provider_code[
            [
                "year",
                "provider_state",
                "rendering_npi",
                "hcpcs_code",
                "provider_code_services",
                "provider_code_payment",
                "prior_provider_code_services",
                "prior_provider_code_payment",
                "service_growth_rate",
                "payment_growth_rate",
            ]
        ],
        on=["year", "provider_state", "rendering_npi", "hcpcs_code"],
        how="left",
    )

    scored["high_payment_volume_flag"] = scored["payment_percentile"].ge(0.90).astype(int)
    scored["high_service_intensity_flag"] = (
        scored["services_per_bene_percentile"].ge(0.90) & scored["services_per_beneficiary"].gt(1.25)
    ).astype(int)
    scored["high_charge_ratio_flag"] = (
        scored["charge_ratio_percentile"].ge(0.90) & scored["charge_to_allowed_ratio"].gt(2.0)
    ).astype(int)
    scored["high_payment_per_bene_flag"] = scored["payment_per_bene_percentile"].ge(0.90).astype(int)
    scored["high_beneficiary_day_flag"] = scored["beneficiary_day_percentile"].ge(0.90).astype(int)
    scored["rapid_growth_flag"] = (
        scored["year"].eq(2024)
        & scored["service_growth_rate"].gt(0.50)
        & scored["payment_growth_rate"].gt(0.50)
        & scored["provider_code_payment"].gt(scored["provider_code_payment"].median())
    ).astype(int)

    scored["audit_priority_score"] = (
        25 * scored["payment_percentile"].fillna(0)
        + 20 * scored["services_per_bene_percentile"].fillna(0)
        + 15 * scored["charge_ratio_percentile"].fillna(0)
        + 15 * scored["payment_per_bene_percentile"].fillna(0)
        + 10 * scored["beneficiary_day_percentile"].fillna(0)
        + 5 * scored["high_charge_ratio_flag"]
        + 5 * scored["high_service_intensity_flag"]
        + 5 * scored["rapid_growth_flag"]
    )

    scored["audit_priority_decile"] = (
        scored.groupby("year")["audit_priority_score"]
        .rank(method="first", pct=True)
        .mul(10)
        .apply(np.ceil)
        .clip(1, 10)
        .astype(int)
    )

    scored.to_csv(PROCESSED_DIR / "claims_audit_scored_records.csv", index=False)

    top_candidates = (
        scored[scored["year"].eq(2024)]
        .sort_values("audit_priority_score", ascending=False)
        .head(250)
        .copy()
    )
    top_candidates.to_csv(TABLE_DIR / "top_2024_audit_candidates.csv", index=False)

    deciles = (
        scored.groupby(["year", "audit_priority_decile"], as_index=False)
        .agg(
            line_count=("rendering_npi", "size"),
            providers=("rendering_npi", "nunique"),
            total_services=("total_services", "sum"),
            estimated_medicare_payment=("estimated_medicare_payment", "sum"),
            high_charge_ratio_flags=("high_charge_ratio_flag", "sum"),
            high_service_intensity_flags=("high_service_intensity_flag", "sum"),
            rapid_growth_flags=("rapid_growth_flag", "sum"),
        )
        .sort_values(["year", "audit_priority_decile"])
    )
    totals = scored.groupby("year", as_index=False).agg(
        all_lines=("rendering_npi", "size"),
        all_payment=("estimated_medicare_payment", "sum"),
        all_services=("total_services", "sum"),
    )
    deciles = deciles.merge(totals, on="year", how="left")
    deciles["line_share"] = deciles["line_count"] / deciles["all_lines"]
    deciles["payment_share"] = deciles["estimated_medicare_payment"] / deciles["all_payment"]
    deciles["service_share"] = deciles["total_services"] / deciles["all_services"]
    deciles.to_csv(TABLE_DIR / "audit_priority_decile_summary.csv", index=False)

    provider_scores = (
        scored[scored["year"].eq(2024)]
        .groupby(
            [
                "rendering_npi",
                "provider_name",
                "provider_state",
                "provider_city",
                "provider_type",
            ],
            as_index=False,
        )
        .agg(
            scored_lines=("hcpcs_code", "size"),
            distinct_hcpcs_codes=("hcpcs_code", "nunique"),
            total_services=("total_services", "sum"),
            estimated_medicare_payment=("estimated_medicare_payment", "sum"),
            mean_audit_priority_score=("audit_priority_score", "mean"),
            max_audit_priority_score=("audit_priority_score", "max"),
            top_decile_lines=("audit_priority_decile", lambda s: int((s == 10).sum())),
            high_charge_ratio_flags=("high_charge_ratio_flag", "sum"),
            high_service_intensity_flags=("high_service_intensity_flag", "sum"),
            rapid_growth_flags=("rapid_growth_flag", "sum"),
        )
        .sort_values(["top_decile_lines", "max_audit_priority_score"], ascending=False)
    )
    provider_scores.to_csv(TABLE_DIR / "provider_audit_score_summary_2024.csv", index=False)
    return scored, deciles, provider_scores


def write_more_tables(scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
    code_family = (
        scored.groupby(["year", "code_family"], as_index=False)
        .agg(
            lines=("hcpcs_code", "size"),
            providers=("rendering_npi", "nunique"),
            services=("total_services", "sum"),
            estimated_medicare_payment=("estimated_medicare_payment", "sum"),
            median_services_per_beneficiary=("services_per_beneficiary", "median"),
            median_charge_to_allowed_ratio=("charge_to_allowed_ratio", "median"),
            top_decile_lines=("audit_priority_decile", lambda s: int((s == 10).sum())),
        )
        .sort_values(["year", "estimated_medicare_payment"], ascending=[True, False])
    )
    code_family.to_csv(TABLE_DIR / "code_family_summary.csv", index=False)

    flag_summary = (
        scored[scored["year"].eq(2024)]
        .groupby("provider_type", as_index=False)
        .agg(
            lines=("hcpcs_code", "size"),
            providers=("rendering_npi", "nunique"),
            estimated_medicare_payment=("estimated_medicare_payment", "sum"),
            high_charge_ratio_flags=("high_charge_ratio_flag", "sum"),
            high_service_intensity_flags=("high_service_intensity_flag", "sum"),
            high_payment_volume_flags=("high_payment_volume_flag", "sum"),
            rapid_growth_flags=("rapid_growth_flag", "sum"),
        )
        .sort_values("estimated_medicare_payment", ascending=False)
    )
    flag_summary.to_csv(TABLE_DIR / "provider_type_flag_summary_2024.csv", index=False)
    return {"code_family": code_family, "flag_summary": flag_summary}


def create_figures(scored: pd.DataFrame, deciles: pd.DataFrame, provider_scores: pd.DataFrame, extra_tables: dict[str, pd.DataFrame]) -> None:
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})

    latest = scored[scored["year"].eq(2024)].copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(latest["audit_priority_score"], bins=30, color="#3b6f8f", edgecolor="white")
    ax.set_title("Distribution of 2024 Audit Priority Scores")
    ax.set_xlabel("Audit priority score")
    ax.set_ylabel("Provider-service records")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "audit_score_distribution_2024.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    latest_deciles = deciles[deciles["year"].eq(2024)].sort_values("audit_priority_decile")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(latest_deciles["audit_priority_decile"], latest_deciles["payment_share"] * 100, color="#6c7f3d")
    ax.axhline(10, color="#444444", linewidth=1, linestyle="--")
    ax.set_title("2024 Payment Share by Audit Priority Decile")
    ax.set_xlabel("Audit priority decile, low to high")
    ax.set_ylabel("Share of estimated Medicare payment")
    ax.set_xticks(range(1, 11))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "payment_capture_by_decile_2024.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    code_family = extra_tables["code_family"]
    latest_family = code_family[code_family["year"].eq(2024)].sort_values("estimated_medicare_payment")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(latest_family["code_family"], latest_family["estimated_medicare_payment"] / 1e6, color="#8a5a44")
    ax.set_title("2024 Estimated Medicare Payment by Code Family")
    ax.set_xlabel("Estimated Medicare payment, millions of dollars")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "payment_by_code_family_2024.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    flag_summary = extra_tables["flag_summary"].head(12).sort_values("high_service_intensity_flags")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(flag_summary["provider_type"], flag_summary["high_service_intensity_flags"], color="#7a3e68")
    ax.set_title("2024 High Service-Intensity Flags by Provider Type")
    ax.set_xlabel("Flagged provider-service records")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "service_intensity_flags_by_provider_type.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    growth = latest.dropna(subset=["service_growth_rate"]).copy()
    growth = growth[(growth["service_growth_rate"] > -1) & (growth["service_growth_rate"] < 5)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(
        growth["service_growth_rate"] * 100,
        growth["payment_growth_rate"] * 100,
        s=np.clip(growth["estimated_medicare_payment"] / 1000, 8, 80),
        alpha=0.35,
        color="#3e6c9a",
        edgecolor="none",
    )
    ax.axvline(50, color="#444444", linewidth=1, linestyle="--")
    ax.axhline(50, color="#444444", linewidth=1, linestyle="--")
    ax.set_title("Provider-HCPCS Growth From 2023 to 2024")
    ax.set_xlabel("Service growth rate")
    ax.set_ylabel("Payment growth rate")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "provider_hcpcs_growth_2024.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    claims: pd.DataFrame,
    scored: pd.DataFrame,
    deciles: pd.DataFrame,
    provider_scores: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    latest = scored[scored["year"].eq(2024)]
    top_decile = deciles[(deciles["year"].eq(2024)) & (deciles["audit_priority_decile"].eq(10))].iloc[0]
    prior = scored[scored["year"].eq(2023)]
    total_payment_2024 = latest["estimated_medicare_payment"].sum()
    total_payment_2023 = prior["estimated_medicare_payment"].sum()
    growth = total_payment_2024 / total_payment_2023 - 1 if total_payment_2023 else np.nan
    top_provider = provider_scores.iloc[0]

    summary = {
        "analytic_records": int(len(scored)),
        "records_2024": int(len(latest)),
        "states": ", ".join(STATES),
        "hcpcs_codes": ", ".join(HCPCS_CODES),
        "providers_2024": int(latest["rendering_npi"].nunique()),
        "estimated_medicare_payment_2024": float(total_payment_2024),
        "estimated_medicare_payment_growth_2023_2024": float(growth),
        "top_decile_line_share": float(top_decile["line_share"]),
        "top_decile_payment_share": float(top_decile["payment_share"]),
        "top_decile_service_share": float(top_decile["service_share"]),
        "top_decile_high_charge_ratio_flags": int(top_decile["high_charge_ratio_flags"]),
        "top_decile_high_service_intensity_flags": int(top_decile["high_service_intensity_flags"]),
        "top_decile_rapid_growth_flags": int(top_decile["rapid_growth_flags"]),
        "top_provider_state": str(top_provider["provider_state"]),
        "top_provider_type": str(top_provider["provider_type"]),
        "top_provider_scored_lines": int(top_provider["scored_lines"]),
        "top_provider_top_decile_lines": int(top_provider["top_decile_lines"]),
        "downloaded_rows": int(quality["rows_downloaded"].sum()),
    }
    with open(OUTPUT_DIR / "analysis_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    text = [
        f"Analytic records: {summary['analytic_records']:,}",
        f"2024 providers: {summary['providers_2024']:,}",
        f"2024 estimated Medicare payment: ${summary['estimated_medicare_payment_2024'] / 1e6:,.2f}M",
        f"Payment growth in selected cohort, 2023-2024: {summary['estimated_medicare_payment_growth_2023_2024'] * 100:,.1f}%",
        f"Top audit-priority decile line share: {summary['top_decile_line_share'] * 100:,.1f}%",
        f"Top audit-priority decile payment share: {summary['top_decile_payment_share'] * 100:,.1f}%",
        f"Top audit-priority decile service share: {summary['top_decile_service_share'] * 100:,.1f}%",
        f"Top decile high charge-ratio flags: {summary['top_decile_high_charge_ratio_flags']:,}",
        f"Top decile high service-intensity flags: {summary['top_decile_high_service_intensity_flags']:,}",
        f"Top decile rapid growth flags: {summary['top_decile_rapid_growth_flags']:,}",
    ]
    (OUTPUT_DIR / "analysis_summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    raw, quality = download_data()
    claims = clean_data(raw)
    write_sqlite(claims)
    scored, deciles, provider_scores = build_audit_scores(claims)
    extra_tables = write_more_tables(scored)
    create_figures(scored, deciles, provider_scores, extra_tables)
    write_summary(claims, scored, deciles, provider_scores, quality)
    print((OUTPUT_DIR / "analysis_summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
