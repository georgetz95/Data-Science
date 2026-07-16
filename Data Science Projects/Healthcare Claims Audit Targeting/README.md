# Healthcare Claims Audit Targeting

This project analyzes selected CMS Medicare Physician and Other Practitioners provider-service records for Indiana and Illinois in 2023 and 2024. The goal is to build a transparent audit-priority score for professional claims review using HCPCS-level utilization, charge, payment, and growth indicators.

## Project Question

Can transparent provider-service features identify a small review cohort that captures a disproportionate share of estimated Medicare payments and service volume?

## Data

- Source: CMS Medicare Physician & Other Practitioners - by Provider and Service
- Years: 2023 and 2024
- States: Indiana and Illinois
- HCPCS cohort: 99203, 99204, 99213, 99214, 99215, 99232, 99285, 93000, 97110
- Analytic records: 237,405 provider-service rows

## Methods

The pipeline downloads public CMS API extracts, cleans provider-service fields, builds a SQLite database, creates SQL summary views, and scores records using:

- payment-volume percentile
- services-per-beneficiary percentile
- charge-to-allowed-ratio percentile
- payment-per-beneficiary percentile
- beneficiary-day intensity percentile
- year-over-year provider-HCPCS service and payment growth

The score is used only as a review-prioritization tool. It does not label records as fraud or improper billing.

## Key Findings

- 2024 cohort: 51,101 providers and about $1.18B in estimated Medicare payments.
- The top audit-priority decile represented 10.0% of 2024 records.
- That decile captured 30.7% of estimated Medicare payments and 31.7% of services.
- The top decile included 2,368 high charge-ratio flags, 7,249 high service-intensity flags, and 2,420 rapid growth flags.

## Main Files

- `scripts/claims_audit_targeting_pipeline.py`: data download, cleaning, SQLite build, scoring, tables, and figures.
- `sql/claims_audit_views.sql`: SQL views for line features, provider profiles, state summaries, and HCPCS summaries.
- `claims_audit_targeting_report.Rmd`: RMarkdown research report.
- `claims_audit_targeting_report.html`: rendered HTML report.
- `latex/claims_audit_targeting_report.tex`: standalone LaTeX paper.
- `latex/claims_audit_targeting_report.pdf`: compiled PDF paper.
- `outputs/tables`: exported quality, summary, decile, provider, and candidate tables.
- `figures`: exported report figures.
