# Medicare Part D Utilization and Cost Modeling

This project analyzes public CMS Medicare Part D Prescribers by Geography and Drug files from 2019 through 2024. The analysis focuses on state-drug records for the 50 states and District of Columbia, with emphasis on cost concentration, utilization features, high-cost prediction, and a simple pharmacy cost-review scenario.

## Research Questions

1. Are CMS drug flags for opioids, long-acting opioids, antibiotics, and antipsychotics associated with cost per beneficiary after adjusting for utilization and demographic/cost-sharing measures?
2. Can prior-year state-drug utilization and cost features identify records that fall in the top decile of next-year cost per beneficiary?
3. How much represented drug cost is concentrated in a targeted review list, and what avoidable-cost reduction would be needed to offset review expense?

## Data

Source: Centers for Medicare & Medicaid Services, Medicare Part D Prescribers by Geography and Drug public-use files.

The workflow uses downloaded CMS CSV files for 2019 through 2024 in `data/raw`. CMS suppression symbols such as `*` and `#` are treated as missing values, not zeros.

## Project Structure

- `scripts/partd_utilization_cost_analysis.R`: main R analysis workflow.
- `scripts/render_report.R`: renders the RMarkdown report to HTML.
- `healthcare_partd_utilization_cost_report.Rmd`: RMarkdown research report.
- `healthcare_partd_utilization_cost_report.html`: rendered HTML report.
- `latex/healthcare_partd_utilization_cost_report.tex`: standalone LaTeX paper source.
- `latex/healthcare_partd_utilization_cost_report.pdf`: compiled LaTeX paper.
- `sql/partd_feature_modeling_template.sql`: warehouse-style SQL feature template.
- `figures/`: exported figures used in the reports.
- `outputs/tables/`: exported analysis tables.
- `outputs/healthcare_partd_summary_deck.pptx`: short summary deck.

## Key Findings

- The analytic file contains 524,753 state-drug-year records after excluding unusable core fields.
- Represented 2024 state-level drug cost totals about $280.8 billion.
- The top 5% of 2024 state-drug records account for 74.5% of represented drug cost.
- A current-cost baseline, logistic regression, and random forest all identify next-year high-cost records with very high discrimination, which points to strong cost persistence.
- The top random forest risk decile captures about 37.2% of represented 2024 drug cost.

## How to Run

From the project folder:

```r
source("scripts/partd_utilization_cost_analysis.R")
source("scripts/render_report.R")
```

The LaTeX paper can be compiled from the `latex` folder with XeLaTeX and BibTeX.
