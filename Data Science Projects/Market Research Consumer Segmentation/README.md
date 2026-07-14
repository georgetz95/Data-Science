# Restaurant Consumer Segmentation and Concept Scoring

**Project type:** Market research analytics, consumer segmentation, concept scorecarding  
**Primary language:** R

## Overview

This project analyzes the UCI Restaurant and Consumer Data set as a market-research style case study. The goal is to show how consumer profile data, stated preferences, venue attributes, and ratings can be joined into a practical workflow for segmentation, multivariate driver analysis, and concept screening.


## Main Question

Which consumer segments are present in the data, which restaurant concept families appear strongest for each segment, and which features remain useful once multiple consumer and venue attributes are modeled together?

## Data Source

The data come from:

Medelln, R. and Serna, J. (2011). *Restaurant and consumer data* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DP41

UCI describes the data as coming from a recommender-system prototype. The raw files include consumer profiles, restaurant attributes, cuisine preferences, payment preferences, and user-restaurant ratings.

## Methods

- Cleaned and joined 9 raw CSV files from the UCI source.
- Engineered market-research style features:
  - cuisine preference match
  - payment method match
  - price within stated budget
  - high satisfaction indicator from the 0 to 2 rating scale
- Built consumer segments using Gower distance and PAM clustering.
- Compared 3-, 4-, and 5-segment solutions using average silhouette width and minimum cluster size.
- Built observed concept scorecards by segment using price tier, alcohol service, and ambience.
- Fit a logistic regression model to estimate multivariate associations with high satisfaction.
- Exported figures, processed data, client tables, and a short PowerPoint summary deck.

## Key Outputs

- `restaurant_consumer_segmentation_report.Rmd`: main project paper.
- `scripts/market_research_analysis.R`: reproducible R pipeline.
- `latex/`: standalone LaTeX source for building the PDF version of the paper.
- `figures/`: final plots used in the report.
- `outputs/client_tables/`: CSV scorecards and model tables.
- `data/processed/`: cleaned modeling data and segment assignments.

## Selected Findings

The final segmentation solution selected 3 consumer groups:

- **S1: Quieter mixed students**
- **S2: Mobile social students**
- **S3: Tech-focused students**

Observed concept scorecards suggest familiar restaurant concepts with stronger service cues tend to perform better. The top overall observed concept is high price, full bar, familiar ambience, with a 52% high-satisfaction rate across 44 ratings.

The logistic model is deliberately more cautious. Several venue effects have confidence intervals that cross 1, which means the scorecards should be used to prioritize follow-up testing rather than make causal claims.

## How to Reproduce

From the project root:

```r
source("scripts/market_research_analysis.R")
source("scripts/render_report.R")
```

The normal workflow uses R Markdown for HTML and the `latex/` folder for the PDF source. Building the LaTeX PDF requires TinyTeX, TeX Live, MiKTeX, or another LaTeX distribution.

## Portfolio Relevance

This project is intended to demonstrate:

- R-based survey and consumer data analysis
- segmentation with mixed data types
- multivariate modeling for satisfaction drivers
- market-research concept scoring
- client-facing communication through charts, tables, and a deck
- reproducible analysis suitable for GitHub
