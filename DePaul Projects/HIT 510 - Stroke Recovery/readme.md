# 🧠 Stroke Recovery & Social Support  
### Causal Inference • IPW • Survey-Weighted Regression • Causal Forests (HTE)

![Project Type](https://img.shields.io/badge/Type-Causal%20Inference-blue.svg)
![Language](https://img.shields.io/badge/R-Programming-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![Dataset](https://img.shields.io/badge/Data-Observational%20Rehabilitation%20Study-orange.svg)

**Authors:** George Tzimas, Samiya Mohsin  
**Date:** May 2025  
**Primary Source:** See project report (PDF included in this repository)

---

## 📌 Project Overview

This project examines whether **perceived social support** causally improves **functional recovery after stroke**, using a U.S. rehabilitation dataset.  
The analysis uses a rigorous causal inference pipeline:

- Inverse Probability Weighting (IPW)
- Survey-weighted regression for ATE estimation
- Causal Forests (GRF) to assess HTE
- Variable importance and calibration tests
- Risk-stratified intervention aligned with CDC HI-5 and Three Buckets of Prevention frameworks

The results highlight patient subgroups that benefit the most from strong social support and inform targeted, equitable stroke rehabilitation strategies.

---

## 📑 Table of Contents

- [Executive Summary](#executive-summary)
- [Introduction](#introduction)
- [Literature Review](#literature-review)
- [Methods](#methods)
- [Results](#results)
- [Discussion](#discussion)
- [Proposed Intervention](#proposed-intervention)
- [Conclusion](#conclusion)
- [Appendices](#appendices)

---

# 🚀 Executive Summary

Using causal inference methods on **1,219** stroke rehabilitation patients, we find:

### ⭐ Key Findings

- **High perceived social support improves functional recovery**
  - **+6.2 FIM** at 3 months  
  - **+7.6 FIM** at 12 months  
  - (IPW-adjusted, p < 0.001)

- **Causal Forests detected significant treatment effect heterogeneity (HTE)**  
  - Calibration differential p < 0.05  
  - Baseline FIM at admission = **top predictor of heterogeneity**

- **Patients with the lowest baseline FIM benefit the most**
  - Q1: **~13.5 FIM points**
  - Q4: ~4.5 FIM points
  - Differences significant (Kruskal–Wallis p < 0.001)

These results support **risk-stratified intervention** for patients with low functional independence at admission.

---

# 📘 Introduction

Stroke remains a leading cause of disability in the United States. While clinical rehabilitation is essential, **social determinants of health**, especially social support, play a crucial role in recovery.

This study evaluates:

1. Whether social support **causally improves** recovery  
2. Which patients benefit **most** (HTE analysis)  
3. How to translate findings into **equitable, data-driven interventions**

---

# 📚 Literature Review

Previous research demonstrates that social support:

- Improves physical function and emotional well-being  
- Reduces depression and social isolation  
- Improves participation and quality of life  
- Maintains and enhances engagement during rehabilitation  

However, few studies use **causal inference with real-world U.S. rehabilitation data**, and even fewer examine **heterogeneous treatment effects**. This project fills that gap.

---

# 🔬 Methods

## 📂 Dataset

- **Study:** Stroke Recovery in Underserved Populations (2005–2006)  
- **Sample Size:** 1,219 patients  
- Contains 226 variables across demographics, health history, comorbidities, facility characteristics, and functional outcomes

---

## 🏁 Outcome Measure

**Functional Independence Measure (FIM)**  
- Primary outcome: **ΔFIM at 3 months**  
- Secondary outcome: **ΔFIM at 12 months**

ΔFIM = FIM_followup − FIM_admission

---

## 💡 Exposure: Social Support

Measured using **Duke–UNC Functional Social Support Questionnaire (FSSQ)**.

- **High Support:** FSSQ ≥ 50  
- **Low Support:** FSSQ < 50  

Binary treatment variable used for causal inference.

---

## 🔧 Covariates (22 Baseline Variables)

- Demographics and socioeconomic factors  
- Comorbidities (HTN, diabetes, arthritis, heart disease, mental health)  
- Stroke severity and length of stay  
- **Baseline FIM at admission (key prognostic factor)**

---

## ⚖️ Modeling Framework

### **1️⃣ Propensity Score Estimation — IPW**

- Logistic regression model predicting probability of high social support  
- Weights:  
  - Treated: 1/e(x)  
  - Control: 1/(1 − e(x))  
- Excellent covariate balance achieved

---

### **2️⃣ IPW Survey-Weighted Regression**

Regression:  
ΔFIM ~ Treatment + Covariates  
Weighted using IPW survey design  
Outcome: ATE at 3 and 12 months

---

### **3️⃣ Causal Forests (HTE Estimation)**

- Implemented using **grf**  
- Predicts individual treatment effects (CATEs)  
- Outputs:  
  - CATE distribution  
  - Calibration test  
  - Variable importance  
  - Subgroup-specific treatment estimates  

Results used to identify **which patients benefit most** from high support.

---

# 📊 Results

## ⭐ IPW-Adjusted Treatment Effect Estimates

| Follow-Up | ATT / ATE (ΔFIM) | p-value |
|-----------|------------------|---------|
| **3 months** | **+6.2** | <0.001 |
| **12 months** | **+7.6** | <0.001 |

High support → substantially better recovery outcomes.

---

## 🎯 Heterogeneous Treatment Effects (HTE)

### CATE by Baseline FIM Quartile

| Quartile | Mean CATE | SD | n |
|----------|-----------|----|---|
| **Q1 (lowest)** | **13.48** | 3.23 | 234 |
| Q2 | 5.24 | 1.39 | 247 |
| Q3 | 4.76 | 0.73 | 216 |
| Q4 (highest) | 4.52 | 0.75 | 231 |

- **Significant heterogeneity:** p < 0.001  
- Lowest baseline FIM → **3× larger benefit**

### Calibration
- Differential term p < 0.05  
- Confirms **meaningful** heterogeneity in treatment effects

### Variable Importance (Top Predictors)
1. **Baseline FIM**
2. Social engagement variables
3. Comorbidities (e.g., diabetes, arthritis)
4. Length of stay / severity metrics

---

# 💬 Discussion

Findings strongly support that:

- Social support **causally improves** stroke recovery  
- The effect is **not uniform** across patients  
- Patients with **low baseline functional status** benefit the most  
- HTE modeling adds actionable insights beyond average treatment effects  
- Results align with decades of rehabilitation research

---

# 🏥 Proposed Intervention  
### *Based on CDC HI-5 & Three Buckets of Prevention*

### **Bucket 1 — Traditional Clinical Care**
- Screen social support using FSSQ at admission  
- Integrate risk flags for low-support / low-FIM patients  
- Prioritize referrals to social workers and peer programs  

### **Bucket 2 — Beyond the Clinic**
- Caregiver training modules  
- Tele-support program for isolated patients  
- Partnerships with community rehab organizations  

### **Bucket 3 — Community / Public Health**
- Awareness campaigns on post-stroke social support  
- Community-based stroke support networks  
- Incorporate social support metrics in county health assessments  

This creates a **scalable, equitable, and evidence-based** intervention pathway.

---

# 🧠 Conclusion

- High social support **significantly improves** functional recovery at 3 and 12 months  
- **Strong HTE detected**, with greatest benefit among patients with low functional independence  
- Supports **risk-stratified**, equity-focused public health interventions  
- Demonstrates the value of combining **IPW, survey regression, and causal forests** on real-world rehab data  

---

# 📚 Appendices

### **Appendix A — Baseline Characteristics**  
Full tables included in report (pages 10–11)

### **Appendix B — IPW Regression Output**  
ATE estimates (pages 12–13)

### **Appendix C — Causal Forest Calibration Results**  
CATE calibration (page 13)

### **Appendix D — Full R Code**  
Complete pipeline (pages 14–48)

---

# 📎 Repository Structure

