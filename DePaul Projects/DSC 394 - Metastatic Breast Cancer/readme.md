# 🩺 Predicting Metastatic Breast Cancer Diagnosis Periods Using Machine Learning  
### WiDS Datathon 2024 • Regression Modeling • Bayesian Modeling • SHAP Analysis

**Authors:**  
- George Tzimas  
- Eric Piatek  
- Esmeralda Villela  
- Sarah Hashmi  

**Course:** DSC 465 – Bayesian Statistics  
**Date:** June 2024  
**Dataset:** HealthVerity – WiDS 2024 University Challenge  

---

## 📌 Table of Contents
- [Executive Summary](#executive-summary)
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Literature Review](#literature-review)
- [Data Overview](#data-overview)
- [Data Preprocessing](#data-preprocessing)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Modeling Methodology](#modeling-methodology)
  - [Bayesian Model](#bayesian-model)
  - [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Feature Importance & SHAP](#feature-importance--shap)
- [Discussion](#discussion)
- [Appendix A: Feature Descriptions](#appendix-a-feature-descriptions)
- [Appendix B: Author Contributions](#appendix-b-author-contributions)

---

# 🚀 Executive Summary
This project applies a combination of **Bayesian regression**, **classical machine learning models**, and **feature importance methods** to predict the **metastatic diagnosis period (days)** for patients with triple-negative breast cancer. Using the **WiDS Datathon 2024 HealthVerity dataset** (over 13,000 patients, 152 features), we examine how:

- socioeconomic factors  
- demographic characteristics  
- climate and ZIP-level variables  
- cancer diagnosis codes  
- and clinical patient data  

affect metastatic diagnosis timelines.

Key results show that:

- **Gradient Boosting Regression (GBR)** is the best-performing model (RMSE ≈ **82**).  
- **Uninformative-prior Bayesian modeling** produces strong and consistent predictions.  
- **Breast cancer diagnosis codes** are the most influential predictors.  
- Patient age, BMI, and regional socioeconomic factors contribute meaningfully.  

---

# 📘 Abstract
Using a comprehensive dataset from the Women in Data Science (WiDS) 2024 University Challenge, we explored machine learning approaches to predict **diagnosis delays in metastatic triple-negative breast cancer**. After extensive exploratory data analysis, missing data imputation, PCA-based dimensionality reduction, regression modeling, and Bayesian simulation using JAGS, we found that ensemble methods—especially GBR—yield strong performance. Feature importance, partial dependence plots, and SHAP values further illuminate how clinical and socioeconomic factors influence diagnostic timelines. These results highlight the promise of ML-driven risk modeling in improving cancer care pathways.

---

# 🩺 Introduction
Breast cancer—particularly **triple-negative breast cancer (TNBC)**—requires rapid identification and treatment. Diagnosis delays worsen survival outcomes, making predictive modeling essential.

This study addresses four central research questions:

1. **Can we accurately predict metastatic diagnosis periods using socioeconomic, demographic, and climate variables?**  
2. **How do socioeconomic factors affect diagnostic timing?**  
3. **How do different missing-data strategies affect model performance?**  
4. **Can Bayesian modeling improve or complement traditional ML-based predictions?**

---

# 📚 Literature Review
The literature review examines global research on breast cancer prediction, ML algorithms, survival modeling, socioeconomic disparities, and the application of Bayesian methods to medical prediction problems.

Studies reviewed include:  
- XGBoost, Random Forests, and neural networks for diagnosis delays (Dehdar et al., 2023)  
- ML survival models (Kourou et al., 2015)  
- Breast cancer recurrence prediction using ML (Lou et al., 2020)  
- High-accuracy SVM-based detection models (Binsaif, 2022)  
- Bayesian networks for prognosis (Choi et al., 2009)  
- Deep learning methods for genomic cancer prognosis (Lee, 2023)

Collectively, these studies highlight the growing value of ML—even though classical models still perform competitively in many cases.

---

# 📊 Data Overview
- **Training set:** 13,173 patients  
- **Test set:** 5,646 patients  
- **Total features:** 152 (141 numeric, 11 categorical)  
- **Target:** `metastatic_diagnosis_period` (days)

Examples of feature types:
- Patient race, gender, age  
- ZIP-level climate and demographic variables  
- Household income and education levels  
- ICD-9 / ICD-10 diagnosis codes  
- BMI and payer type  
- Population density  

---

# 🧼 Data Preprocessing

### ✔ Missing Data Imputation
- BMI missingness: ~51%  
- Race missingness: ~37%  
- Mean/mode imputation insufficient → **Iterative multivariate imputation** used.  
- Features missing 100% (`metastatic_first_novel_treatment`) were dropped.  

### ✔ Outlier Removal
BMI values above 50 were removed as extreme outliers.

---

# 📉 Dimensionality Reduction

Using **Principal Component Analysis (PCA)**:

- 136 socioeconomic/climate features reduced to a single principal component (**PC1**)
- PC1 captured **99% of the variance**
- Final dataset reduced from 152 → **12 total features**

This drastically simplifies modeling while retaining essential information.

---

# 🧠 Modeling Methodology

## **Bayesian Model**
Developed using **JAGS + CODA**, with:

- 4 chains  
- 20,000 iterations per chain  
- Burn-in = 10,000  
- Both **informative** and **uninformative priors** tested  
- Variables included: PC1, patient age, encoded diagnosis codes, state-level mean encodings  

### Key Bayesian Findings:
- **Uninformative priors outperformed informative priors**  
  - RMSE: **86.79** (Model 2, 1,000 samples)  
- Diagnostic checks (trace plots, Geweke scores, ESS) confirmed strong convergence  

---

## **Machine Learning Models**
Eight regression models were tested:

| Model | RMSE | MAE | R² |
|-------|------|------|------|
| **Gradient Boosting (GBR)** | **82.78** | 64.40 | **0.418** |
| MLP Regressor | 82.86 | 62.73 | 0.417 |
| CatBoost | 82.90 | 63.20 | 0.416 |
| Random Forest | 84.78 | 64.26 | 0.390 |
| XGBoost | 85.31 | 64.78 | 0.382 |
| Bayesian Regression | 86.79 | 66.91 | – |
| AdaBoost | 89.45 | 76.53 | 0.321 |
| SVR | 109.76 | 77.22 | -0.023 |

### Hyperparameter tuning
A 3,888-run GridSearchCV identified the optimal GBR parameters.

**Final GBR Test Performance**
- **MSE:** 6737  
- **RMSE:** 82.08  
- **MAE:** 62.85  
- **R²:** 0.419  

---

# 🔍 Feature Importance & SHAP

### **Most important features**
1. **Breast cancer diagnosis codes (ICD-9 / ICD-10)**  
2. **Patient age**  
3. **BMI**  
4. **ZIP3 region**  
5. **PC1 (socioeconomic composite)**  

### SHAP findings
- Diagnosis codes have the strongest positive effect on predicted delay.  
- Age shows nonlinear effects—certain ranges exhibit abrupt changes in diagnosis period.  
- BMI influences predictions weakly but meaningfully in interaction terms.  
- Location variables matter more through socioeconomic clustering than direct effects.  

---

# 💬 Discussion
Our findings demonstrate:

- Ensemble ML models—especially GBR—excel at predicting diagnosis delays.  
- Bayesian modeling is a strong complementary method, especially for uncertainty estimation.  
- Diagnosis codes reveal strong clinical signals affecting diagnostic timelines.  
- Socioeconomic variables (income, education, insurance type) matter but have subtle effects.  

Machine learning offers promising tools for identifying at-risk patients and improving early diagnosis pathways for metastatic breast cancer.

---

# 📎 Appendix A: Feature Descriptions
A complete feature dictionary (over 150 variables) is included, covering:

- Demographic and patient-level attributes  
- Insurance information  
- ZIP3-level socioeconomic indicators  
- Population and climate features  
- Medical diagnosis codes  

---

# 👥 Appendix B: Author Contributions

**George Tzimas**  
- Missing data imputation  
- PCA dimensionality reduction  
- Regression model testing  
- Hyperparameter tuning  
- Feature importance + SHAP  

**Eric Piatek**  
- Categorical encoding strategy  
- Stepwise regression analysis  
- Full Bayesian model development  
- Convergence diagnostics  

**Esmeralda Villela**  
- Introduced WiDS project concept  
- Preliminary EDA and regression models  
- Random Forest, XGBoost, Ridge Regression  

**Sarah Hashmi**  
- Correlation matrices and EDA visualizations  
- Presentation preparation  
- Literature review  

---

## 📫 Contact
For questions or collaboration:

**George Tzimas**  
📧 georgetz95@gmail.com  
🔗 GitHub: https://github.com/georgetz95  
🔗 LinkedIn: https://www.linkedin.com/in/georgetz95

