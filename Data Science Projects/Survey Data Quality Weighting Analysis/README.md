# Survey Data Quality and Weighting Analysis

This project analyzes public NHANES 2017-2018 survey data to evaluate whether health insurance coverage and having a usual source of care are associated with fair or poor self-rated health among U.S. adults.

The project emphasizes survey-data workflow skills: importing SAS transport files, joining person-level questionnaire files, building a variable dictionary, checking missingness and coding validity, applying interview sample weights, fitting survey-weighted models, and producing report-ready tables and figures.

## Research Question

Among adults aged 18 and older, are insurance status and usual source of care associated with fair or poor self-rated health after accounting for age, gender, race/ethnicity, education, and family income to poverty ratio?

## Data

Public-use files come from the National Health and Nutrition Examination Survey 2017-2018 cycle:

- `DEMO_J`: demographics, interview weights, strata, and masked PSUs
- `HUQ_J`: health status, health care utilization, and access to care
- `HIQ_J`: health insurance coverage

## Project Structure

- `scripts/01_prepare_nhanes_survey_analysis.R`: main R pipeline for data preparation, quality checks, survey-weighted estimates, imputation, modeling, and figures
- `scripts/02_python_quality_model_check.py`: Python QA and train/test model validation companion script
- `scripts/03_sas_transport_survey_workflow.sas`: SAS companion workflow for XPT import, recoding, survey frequencies, and survey logistic modeling
- `data/raw`: downloaded NHANES SAS transport files
- `data/processed`: analytic CSV output
- `outputs`: QA tables, variable dictionary, estimates, and model results
- `figures`: exported figures
- `paper`: R Markdown report source and rendered HTML
- `latex`: standalone LaTeX report source and PDF

