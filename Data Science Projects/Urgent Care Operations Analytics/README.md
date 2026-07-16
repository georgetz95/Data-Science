# Urgent Care Operations Analytics

This project analyzes selected CMS Medicare Physician and Other Practitioners provider-service records for Illinois and Indiana in 2023 and 2024. The goal is to show how clinic-relevant service data can be cleaned, summarized, and ranked for operational review.

## Project Question

Can transparent operational features identify a small review cohort that captures a disproportionate share of service volume and estimated payment?

## Data

- Source: CMS Medicare Physician & Other Practitioners - by Provider and Service
- Years: 2023 and 2024
- States: Illinois and Indiana
- HCPCS cohort: 99203, 99204, 99213, 99214, 99215, 99232, 99285, 93000, 97110
- Analytic records: 237,405 provider-service rows

The R pipeline uses the CMS extract already downloaded in the healthcare claims project when it is available. If that local extract is not present, the script can download the selected CMS slices directly from the CMS API.

## Methods

The pipeline reads public CMS provider-service records, cleans identifiers and numeric fields, creates service-family and market summaries, builds a SQLite database, exports SQL-ready views, and scores records using:

- service volume
- estimated Medicare payment
- services per beneficiary
- charge-to-allowed ratio
- payment per service
- year-over-year service and payment growth

The score is used as an operational review-prioritization tool. It is not a clinical risk score, compliance score, or claim-level adjudication model.

## Key Findings

- 2024 cohort: 51,101 providers, 861 city-state markets, and 18.7M selected services.
- 2024 estimated Medicare payment: about $1.18B.
- The top operational-priority decile represented 10.0% of 2024 records.
- That decile captured 40.0% of selected services and 42.1% of estimated Medicare payments.
- Chicago, IL was the largest market in the extract, with about 1.99M selected services.

## Main Files

- `scripts/clinic_operations_pipeline.R`: data preparation, scoring, tables, figures, and SQLite build.
- `sql/clinic_operations_views.sql`: SQL views for service-family, market, provider-type, and priority-decile summaries.
- `urgent_care_operations_report.Rmd`: RMarkdown research report.
- `urgent_care_operations_report.html`: rendered HTML report.
- `latex/urgent_care_operations_report.tex`: standalone LaTeX paper.
- `latex/urgent_care_operations_report.pdf`: compiled PDF paper.
- `outputs/tables`: exported summary and review-priority tables.
- `figures`: exported report figures.
- `application_materials`: one-page resume and cover letter for the healthcare data analyst role.
