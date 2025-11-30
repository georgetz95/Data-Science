# 🎯 Estimating Heterogeneous Treatment Effects Using Hierarchical Bayesian Models  
### Hierarchical Bayesian Modeling • HTE • Causal Inference • Simulation Study

**Author:** George Tzimas  
**Date:** November 2024  
**Source:** Full project paper included in repository (PDF)

---

## 📑 Table of Contents
- [Executive Summary](#executive-summary)
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Methods](#methods)
  - [Synthetic RCT Data Generation](#synthetic-rct-data-generation)
  - [Outcome Model](#outcome-model)
  - [Hierarchical Bayesian Model](#hierarchical-bayesian-model)
- [Results](#results)
  - [Model Diagnostics](#model-diagnostics)
  - [CATE Estimation Performance](#cate-estimation-performance)
  - [Risk-Stratified Treatment Effects](#risk-stratified-treatment-effects)
- [Discussion](#discussion)
- [Limitations](#limitations)
- [Appendix A – Data Generation Code](#appendix-a--data-generation-code)
- [Appendix B – JAGS Model](#appendix-b--jags-model)

---

# 🚀 Executive Summary
This project develops a **three-level hierarchical Bayesian logistic regression model** to estimate **heterogeneous treatment effects (HTE)** in a simulated randomized clinical trial. Synthetic patient-level data were generated to mimic realistic clinical covariates—including age, BMI, blood pressure, comorbidities, smoking status, and prior bleeding history—and a binary outcome representing major bleeding.

The Bayesian model successfully recovers the underlying true treatment effects embedded in the simulation:

- **Correlation between predicted and true CATEs:** **0.78**  
- **RMSE:** **0.53**  
- **95% credible interval coverage:** **99.8%**

Risk-stratified results show large differences in treatment benefit. Low-risk participants experience potential harm (CATE < 0), while high-risk participants show strong benefit. This demonstrates the value of Bayesian hierarchical approaches for personalized medicine and treatment effect estimation.

---

# 📘 Abstract
Randomized clinical trials (RCTs) traditionally focus on estimating average treatment effects (ATE). However, treatment response can vary considerably across patient subgroups. This study develops a **hierarchical Bayesian model** implemented via **JAGS** to estimate heterogeneous treatment effects using fully simulated RCT data.

Synthetic covariates (age, blood pressure, BMI, diabetes, anticoagulant use, prior bleeding, smoking status) were generated to reflect realistic clinical relationships. A binary bleeding event served as the main outcome.

The model captured individual-level variability with strong alignment to the true data-generating process (correlation = 0.78, RMSE = 0.53). Subgroup analysis showed treatment harm in low-risk patients and substantial benefit in high-risk groups. While performance is promising, future work should extend to real clinical datasets to assess generalizability.  
(:contentReference[oaicite:1]{index=1})

---

# 🩺 Introduction

## Background
Average treatment effects alone may obscure clinically important **heterogeneity**. Understanding differential treatment responses is essential for:

- Precision medicine  
- Personalized risk-benefit assessment  
- Treatment selection  
- Subgroup-specific clinical guidance  

## Objectives
This project aims to:

1. Generate synthetic RCT data with realistic covariates and heterogeneous treatment effects.  
2. Build a hierarchical Bayesian model to estimate individual-level and subgroup-level treatment effects.  
3. Validate how well the model recovers the true treatment effect structure.  

(:contentReference[oaicite:2]{index=2})

---

# 🔬 Methods

## Synthetic RCT Data Generation
Covariate distributions were designed to resemble real-world clinical data (Table 2, page 2):

- **Age:** Normal (55, SD 12), truncated 25–85  
- **BMI:** Normal (28, SD 5), truncated 16–50  
- **Blood pressure:** Multivariate normal  
- **Comorbidities:**  
  - Hypertension  
  - Diabetes  
  - Smoking  
  - Anticoagulant use  
  - Prior bleeding  
- **Treatment:** Randomized 50/50  
- **Sample size:** ~2000  

Baseline risk is generated using logistic regression (Eq. 1–2, page 2). Treatment effects include interactions:

- Age > 65  
- Anticoagulant use  
- High BMI  
(:contentReference[oaicite:3]{index=3})

---

## Outcome Model
The probability of a bleeding event combines:

### Baseline risk  
via logistic regression (age, diabetes, anticoagulant use, SBP, prior bleeding)

### Treatment effect  
via heterogeneous modifiers:

- Older adults → more harm  
- Anticoagulant users → more harm  
- BMI > 30 → more harm  

### Final event probability:
\[
\text{logit}(P_{\text{event}}) = \text{logit}(P_{\text{baseline}} + T_i \cdot TE_i)
\]
(:contentReference[oaicite:4]{index=4})

---

## Hierarchical Bayesian Model
Three-level structure (Section 2.2, page 5):

1. **Population-level effects:**  
   Fixed effects for baseline covariates and treatment.

2. **Subgroup-level random effects:**  
   Gender-specific random intercepts and random treatment effects.

3. **Individual-level deviations:**  
   Capture within-subgroup heterogeneity.

### Likelihood: Logistic regression  
### Priors:  
- All fixed effects: Normal(0,1)  
- Random effect precisions: Gamma(2,2)  
(:contentReference[oaicite:5]{index=5})

---

# 📊 Results

## Model Diagnostics
Convergence validated using:

- **Gelman–Rubin PSRF:** All ≈ **1.00–1.04** (Table 5)  
- **Geweke z-scores:** All within ±2 (Table 6)  
- Trace plots show good mixing (Figure 5, page 14)  
- Posterior densities smooth and well-behaved (Figure 6)  
(:contentReference[oaicite:6]{index=6})

---

## CATE Estimation Performance
Comparison of true vs. predicted CATEs:

| Metric | Value |
|--------|--------|
| **Correlation** | **0.784** |
| **RMSE** | **0.529** |
| **MAE** | **0.413** |
| **95% CI Coverage** | **0.998** |

(Table 7, page 8)  
(:contentReference[oaicite:7]{index=7})

---

## Risk-Stratified Treatment Effects
Participants were grouped into five equal-sized baseline-risk strata (Table 8, page 8).

### Key Findings:
- **Q1 (lowest risk):** Treatment harmful (mean TE = -0.359)  
- **Q2:** Minimal harm  
- **Q3:** Mild benefit  
- **Q4:** Strong benefit  
- **Q5 (highest risk):** **Strongest benefit** (mean TE = 0.547)  

This shows a **monotonic increase in benefit** with baseline risk.

Visualizations:

- Distribution of predicted CATEs (Figure 2)  
- CATEs by strata (Figure 3)  
- Individual-level risk curves (Figure 4)  
(:contentReference[oaicite:8]{index=8})

---

# 💬 Discussion
This study demonstrates that:

- Bayesian hierarchical models can **accurately recover treatment effect heterogeneity**.  
- Substantial differences in treatment response occur across risk groups.  
- High-risk individuals benefit most, while low-risk individuals may be harmed.  
- This aligns with principles of personalized medicine and individualized treatment decisions.  
(:contentReference[oaicite:9]{index=9})

---

# ⚠️ Limitations
- Data are fully simulated—real-world noise and confounding may reduce performance.  
- Only a limited set of covariates was included relative to real clinical trials.  
- No time-to-event modeling (e.g., Bayesian survival models).  
- More complex interactions and nonlinearities may require advanced models (e.g., BART, Bayesian causal forests).  
(:contentReference[oaicite:10]{index=10})

---

# 📎 Appendix A – Data Generation Code
Full R function used to simulate the dataset (pages 10–12).  
Includes generation of covariates, baseline risk, treatment interactions, and event probabilities.  
(:contentReference[oaicite:11]{index=11})

---

# 📎 Appendix B – JAGS Model
Complete hierarchical Bayesian model specification (pages 12–13).  
Includes:

- Likelihood  
- Fixed effects  
- Interaction effects  
- Random effects for gender  
- Hyperpriors  
- Gender-specific treatment effect calculation  
(:contentReference[oaicite:12]{index=12})

---

# 📫 Contact
For questions or collaboration:

**George Tzimas**  
📧 georgetz95@gmail.com  
🔗 GitHub: https://github.com/georgetz95  
🔗 LinkedIn: https://www.linkedin.com/in/georgetz95
