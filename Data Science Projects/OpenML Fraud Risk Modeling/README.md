# Fraud Risk Modeling

This portfolio project emphasizes fraud and identity risk, Python 3, PostgreSQL-style feature engineering, cloud-oriented infrastructure design, full model-lifecycle ownership, production-ready code, data quality, feature engineering, and monitoring.

The analysis uses the OpenML credit card fraud dataset, a two-day sample of September 2013 European card transactions with 284,807 transactions and 492 fraud cases. The modeling task is framed like a fraud operations problem: rank transactions by risk, choose a threshold from a validation review budget, and monitor score and feature drift in the later test window.

## Main Result

The selected balanced random forest achieved a test average precision of 0.778. At a validation-tuned 1% review budget, the model alerted 395 of 42,722 test transactions, found 44 of 52 fraud cases, and reached 84.6% recall with 11.1% precision.

## Folder Guide

- `scripts/fraud_risk_modeling.py`: Python analysis script for data preparation, feature engineering, model fitting, evaluation, plots, monitoring tables, and SQLite exports.
- `sql/postgres_feature_scoring_template.sql`: Feature table template for a production-style warehouse workflow.
- `references.bib`: bibliography used by the RMarkdown report.
- `figures/`: generated figures for class balance, precision-recall, review budget capture, score distribution, drift, and monitoring.
- `outputs/tables/`: CSV outputs used in the report.
- `outputs/fraud_risk_scoring.sqlite`: local scoring database with feature, score, threshold, and monitoring tables.
