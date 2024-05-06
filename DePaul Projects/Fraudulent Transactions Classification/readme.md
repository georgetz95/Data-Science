# Predicting Fraudulent Transactions Using Binary Classification Algorithms

**Team Member:** Giorgos Tzimas  
**Date:** May 2023  

## Abstract
Using customers’ transactional dataset, we built two supervised machine learning algorithms, Logistic Regression and Random Forest, to classify each instance as either a fraudulent or non-fraudulent transaction. After comparing the performance of both models on important metrics such as recall and False Positive Rate, we picked Random Forest as the final binary classification model to be used.

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Background](#background)
  - [Observations and Attributes](#observations-and-attributes)
  - [Data Types](#data-types)
- [Data Cleaning](#data-cleaning)
- [Feature Extraction](#feature-extraction)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Total Counts by Fraud Status](#total-counts-by-fraud-status)
  - [Numeric Feature Distribution](#numeric-feature-distribution)
  - [Numeric Feature Boxplot by Fraud Status](#numeric-feature-boxplot-by-fraud-status)
  - [Transaction Amount by Fraud Status](#transaction-amount-by-fraud-status)
  - [Transactions by Country](#transactions-by-country)
  - [Fraudulent Transaction Amount by Country](#fraudulent-transaction-amount-by-country)
  - [Transactions by Card Presence](#transactions-by-card-presence)
  - [Top 20 Brands by Transaction Count](#top-20-brands-by-transaction-count)
  - [Top 20 Brands by Fraudulent Transaction Count](#top-20-brands-by-fraudulent-transaction-count)
  - [Top 20 Brands by Average Fraudulent Transaction Amount](#top-20-brands-by-average-fraudulent-transaction-amount)
- [Unsupervised Learning](#unsupervised-learning)
  - [Dimensionality Reduction: Singular Value Decomposition](#dimensionality-reduction-singular-value-decomposition)
  - [Clustering: KMeans](#clustering-kmeans)
- [Supervised Learning](#supervised-learning)
  - [Dataset Resampling](#dataset-resampling)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Logistic Regression](#logistic-regression)
    - [Feature Importances](#logistic-regression-feature-importances)
  - [Random Forest](#random-forest)
    - [Feature Importances](#random-forest-feature-importances)
- [Model Comparison](#model-comparison)
- [References](#references)