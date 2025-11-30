# 🩺 Liver Cirrhosis Survival Analysis  
### Logistic Regression • Cox Proportional Hazards • Kaplan–Meier • Mixed-Type Correlations

![Project Type](https://img.shields.io/badge/Type-Statistical%20Modeling-blue.svg)
![Language](https://img.shields.io/badge/R-Programming-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![Dataset](https://img.shields.io/badge/Data-Mayo%20Clinic%20RCT-orange.svg)

**Author:** George Tzimas  
**Date:** March 2025  
**Primary Source:** See project report (included in repository)

---

## 📌 Project Overview

This project investigates **predictors of mortality in primary biliary cirrhosis** using:

- Logistic Regression  
- Cox Proportional Hazards Model  
- Kaplan–Meier Survival Analysis  
- Mixed-type correlation matrices  
- Model calibration and risk stratification  

The analysis uses data from a historical **Mayo Clinic randomized controlled trial** evaluating D-penicillamine, updated with modern statistical modeling techniques.

---

## 📑 Table of Contents

- [Executive Summary](#executive-summary)
- [Abstract](#abstract)
- [Methods](#methods)
- [Results](#results)
- [Discussion](#discussion)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [Appendices](#appendices)

---

# 🚀 Executive Summary

This study evaluates **clinical, demographic, and biochemical factors** associated with mortality due to liver cirrhosis. Using both logistic regression and Cox survival modeling, we identify strong predictors including:

- Bilirubin  
- Albumin  
- Prothrombin time  
- SGOT  
- Copper levels  
- Age  
- Edema and Ascites  
- Stage of cirrhosis  

### ⭐ Key Performance Metrics

| Model | Best Metric |
|-------|-------------|
| Logistic Regression | **AUC = 0.89** |
| Cox Proportional Hazards | **C-index = 0.847** |

**D-penicillamine showed no significant survival benefit**, confirmed via log-rank test and Cox modeling.

---

# 📘 Abstract

This study applies logistic regression and Cox proportional hazards modeling to identify predictors of mortality among patients with liver cirrhosis. Missing values were imputed using the *missForest* algorithm. Stepwise model selection identified bilirubin, albumin, prothrombin, SGOT, copper, and age as significant predictors. Both models align with established hepatology literature, confirming their prognostic relevance.

---

# 🔬 Methods

## 📂 Data Source

- Mayo Clinic randomized controlled trial, 1974–1984  
- Publicly available survival dataset (312 PBC patients, 19 variables)  
- Contains demographics, biochemical markers, clinical characteristics, treatment assignment, and survival time  

---

## 🧼 Preprocessing

- Removed transplanted patients → final **n = 393**  
- Converted categorical variables to factors (Stage, Edema, Ascites, Hepatomegaly, Sex)  
- Missing values imputed using **missForest** with OOB error **0.01**  
- Event outcome recoded: **0 = Censored**, **1 = Death**  

---

## 📊 Exploratory Data Analysis

EDA included:

- Mixed-type correlation heatmap  
- Event rates by sex, stage, and treatment  
- Boxplots (e.g., age by gender)  
- Kaplan–Meier survival curves with risk tables  

Findings:

- Strong correlations for bilirubin, prothrombin, copper, and SGOT  
- Mortality increases sharply by disease stage  
- No survival difference between treatment arms (log-rank p = 0.79)  

---

## 📈 Statistical Models

### **1️⃣ Logistic Regression**

- Built using all baseline predictors  
- Stepwise AIC selection  
- Assessed via AUC, ROC curves, calibration, and Hosmer–Lemeshow test  

### **2️⃣ Cox Proportional Hazards Model**

- Evaluates impact of predictors on time-to-death  
- Proportional hazard assumptions verified via Schoenfeld residuals  
- C-index ≈ **0.847**  

---

# 📊 Results

## Logistic Regression Findings

Significant predictors (OR > 1 unless noted):

- Ascites (OR ~9.8)  
- Hepatomegaly (OR ~2.0)  
- Bilirubin (OR 1.21 per mg/dL)  
- Copper (OR 1.006 per µg)  
- Alkaline phosphatase (significant small effect)  
- SGOT (OR 1.007 per U/L)  
- Prothrombin time (OR 1.85 per second)  
- Age (OR 1.047 per year)  

**AUC ≈ 0.89**

---

## Cox Proportional Hazards Findings

| Predictor | Hazard Ratio | Interpretation |
|----------|--------------|----------------|
| Edema (Y) | **2.39** | Major risk factor |
| Bilirubin | **1.087** | Strong biochemical predictor |
| Albumin | **0.456** | Protective |
| Copper | **1.003** | Small but significant |
| SGOT | **1.004** | Indicates liver damage |
| Prothrombin | **1.359** | Clotting dysfunction → risk |
| Stage L | **4.91** | Drastically higher hazard |
| Age | **1.029** | Older age → higher risk |

Treatment effect not significant (p = 0.79).

---

# 💬 Discussion

The results match well-established findings in hepatology:

- **Bilirubin** and **prothrombin time** reflect impaired liver clearance and synthetic function  
- **Low albumin** indicates severe hepatic dysfunction  
- **Edema and ascites** reflect portal hypertension and decompensation  
- **Age** remains a strong non-modifiable risk factor  

The absence of treatment effect from D-penicillamine is consistent with historical clinical evidence.

---

# ⚠️ Limitations

- Sample size ~400 limits external generalizability  
- Dataset reflects clinical practices from the 1970s–80s  
- Some mild deviations in PH assumptions  
- No external validation cohort available  

---

# 🧠 Conclusion

Both statistical models consistently highlight bilirubin, albumin, prothrombin, SGOT, copper, and age as clinically meaningful predictors of mortality. Future research should incorporate modern, larger, and more diverse cohorts to better understand long-term survival patterns and treatment effects.

---

# 📚 Appendices

## Appendix A — Tables
Includes:
- Baseline characteristics  
- Missing value summary  
- Logistic regression coefficients  
- Cox model coefficients  
- Schoenfeld residual tests  

## Appendix B — Figures
Includes:
- Correlation heatmap  
- EDA barplots  
- Kaplan–Meier survival curves  
- Calibration plots  
- Hazard ratio plots  

## Appendix C — Full R Code
Includes:
- Data cleaning  
- Missing data imputation  
- Logistic regression  
- Cox modeling  
- Calibration analysis  
- Mixed-type correlation matrix function  
- Risk stratification and subgroup analysis  

---

## 📎 Repository Structure

