# Student Support Dashboard Analytics

This project analyzes public education data for Maine Township High School District 207 in Park Ridge, Illinois. The goal is to turn school-level attendance, behavior, enrollment, subgroup, and data-quality records into a district-style monitoring workflow that could support student support planning, dashboard review, and follow-up questions for building leaders.

The analysis uses public data available through the Urban Institute Education Data Portal, including Common Core of Data school directory/enrollment records and Civil Rights Data Collection records. It focuses on school-level patterns because the public data do not contain student-level intervention assignments, course records, or personally identifiable student information.

## Research Questions

1. Can public school data be integrated into a clean district monitoring layer for attendance, behavior, enrollment, and subgroup review?
2. Do school-level attendance and behavior indicators identify different support-priority profiles across high schools?
3. Which subgroup patterns should be flagged for deeper local review, while respecting small-denominator limits?

## Project Structure

- `scripts/01_download_education_data.R`: downloads CCD and CRDC records from the Urban Institute Education Data Portal API.
- `scripts/02_analyze_student_support.R`: cleans public records, builds rates and priority scores, exports processed tables and figures.
- `scripts/03_create_sqlite_reporting.py`: loads processed CSVs into SQLite and creates dashboard-ready SQL views.
- `sql/student_support_reporting_views.sql`: reporting views for KPI, trend, subgroup, behavior, MTSS-priority, and QA tables.
- `paper/student_support_analytics_report.Rmd`: research-style RMarkdown paper.
- `paper/student_support_analytics_report.html`: rendered HTML paper.
- `latex/student_support_analytics_report.tex`: standalone LaTeX paper source.
- `latex/student_support_analytics_report.pdf`: rendered PDF from the LaTeX source.
- `dashboard/student_support_dashboard.Rmd`: compact HTML dashboard.
- `dashboard/student_support_dashboard.html`: rendered dashboard.
- `dashboard/tableau_dashboard_spec.md`: Tableau-style dashboard worksheet and data-source notes.
- `application_materials/`: resume and cover letter files.

## Main Outputs

- Latest active CCD enrollment across the three comprehensive high schools: 6,272 students.
- District 2021 chronic absenteeism rate from public CRDC totals: 34.1%.
- District 2021 suspension instances per 100 students: 4.6.
- Maine West High School ranked highest on the composite support-priority score, followed by Maine East High School and Maine South High School.
- Subgroup review flags use denominator thresholds so very small public-data cells are not overinterpreted.

## How To Run

From the project folder:

```r
source("scripts/01_download_education_data.R")
source("scripts/02_analyze_student_support.R")
rmarkdown::render("paper/student_support_analytics_report.Rmd")
rmarkdown::render("dashboard/student_support_dashboard.Rmd")
```

Then create the SQLite reporting layer:

```bash
python scripts/03_create_sqlite_reporting.py
```

The LaTeX paper is compiled from `latex/student_support_analytics_report.tex`.
