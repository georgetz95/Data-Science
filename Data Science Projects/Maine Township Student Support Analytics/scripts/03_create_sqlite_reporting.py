from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_DIR = Path.cwd()
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
SQL_PATH = PROJECT_DIR / "sql" / "student_support_reporting_views.sql"
DB_PATH = PROCESSED_DIR / "student_support_reporting.sqlite"


TABLES = {
    "school_directory_2018_2024": "school_directory_2018_2024.csv",
    "school_directory_latest": "school_directory_latest.csv",
    "chronic_absenteeism_school_trend": "chronic_absenteeism_school_trend.csv",
    "subgroup_chronic_absenteeism_2021": "subgroup_chronic_absenteeism_2021.csv",
    "discipline_instances_school_trend": "discipline_instances_school_trend.csv",
    "mtss_priority_school_table": "mtss_priority_school_table.csv",
    "dashboard_kpis": "dashboard_kpis.csv",
    "data_quality_checks": "data_quality_checks.csv",
}

EXPORTS = {
    "vw_dashboard_kpis": "sql_dashboard_kpis.csv",
    "vw_chronic_absence_school_trend": "sql_chronic_absence_school_trend.csv",
    "vw_subgroup_equity_review": "sql_subgroup_equity_review.csv",
    "vw_behavior_support_monitoring": "sql_behavior_support_monitoring.csv",
    "vw_mtss_priority_queue": "sql_mtss_priority_queue.csv",
    "vw_data_quality_checks": "sql_data_quality_checks.csv",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as con:
        for table_name, csv_name in TABLES.items():
            df = pd.read_csv(PROCESSED_DIR / csv_name)
            df.to_sql(table_name, con, if_exists="replace", index=False)

        with open(SQL_PATH, "r", encoding="utf-8") as sql_file:
            con.executescript(sql_file.read())

        for view_name, export_name in EXPORTS.items():
            df = pd.read_sql_query(f"SELECT * FROM {view_name}", con)
            df.to_csv(OUTPUT_DIR / export_name, index=False)

    print(f"SQLite reporting database saved to {DB_PATH}")


if __name__ == "__main__":
    main()
