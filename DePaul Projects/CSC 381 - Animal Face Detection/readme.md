# 🐾 Analysis of Image Feature Extraction Techniques for Animal Face Detection
### HOG • Sobel • Laplacian • SVC Classification • Computer Vision • Flask Deployment

**Author:** George Tzimas  
**Date:** November 2024  
**Source:** Full project report included in repository (PDF)

---

## 📑 Table of Contents
- [Executive Summary](#executive-summary)
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Background & Prior Work](#background--prior-work)
- [Methods](#methods)
  - [Dataset](#dataset)
  - [Preprocessing](#preprocessing)
  - [Feature Extraction Methods](#feature-extraction-methods)
  - [Model Training & Evaluation](#model-training--evaluation)
- [Results](#results)
- [Model Deployment](#model-deployment)
- [Conclusion](#conclusion)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Appendix – Classification Metrics](#appendix--classification-metrics)
- [References](#references)

---

# 🚀 Executive Summary
This project evaluates three traditional computer vision feature extraction techniques—**Histogram of Oriented Gradients (HOG)**, **Sobel**, and **Laplacian**—to classify animal faces from the **LHI Animal Faces dataset** containing 20 classes (19 animal species + humans).

All extracted features were used to train a **Support Vector Classifier (SVC)** and compared across precision, recall, F1 score, and accuracy. HOG features produced the strongest results across nearly all classes, outperforming the Sobel and Laplacian approaches by a wide margin.

The final pipeline includes:

- Preprocessing (grayscale → histogram equalization → Gaussian noise)  
- HOG feature extraction  
- Hyperparameter-tuned SVC model (accuracy: **0.746**)  
- **Flask web application** for interactive animal face classification  

(:contentReference[oaicite:1]{index=1})

---

# 📘 Abstract
This study investigates traditional feature extraction methods for multi-class image classification using the LHI Animal Faces dataset (20 classes). Three methods—HOG, Sobel, and Laplacian—were implemented to extract structural and textural information. Each feature set was used to train an SVC model and evaluated using precision, recall, and F1 score.

HOG outperformed the other methods due to its strong ability to capture shape and gradient information. A tuned SVC model trained on HOG features achieved an F1 score of **0.747**. The full pipeline was deployed as a Flask web app to allow interactive predictions.  
(:contentReference[oaicite:2]{index=2})

---

# 🩺 Introduction
Modern image classification systems often rely on deep learning (e.g., CNNs such as AlexNet, VGG, and ResNet). However, these models require large datasets, substantial training time, and lack interpretability.

Traditional feature extraction methods like HOG, Sobel, and Laplacian offer:

- Greater interpretability  
- Lower data requirements  
- Faster computation  
- Compatibility with classical ML models  

This project compares the performance and trade-offs of these traditional approaches when classifying animal faces.  
(:contentReference[oaicite:3]{index=3})

---

# 📚 Background & Prior Work
Key literature supporting this work:

- **Dalal & Triggs (2005):** Established HOG as a robust descriptor for object detection.  
- **Rangdal & Hanchate (2014):** Applied HOG to animal detection tasks.  
- **Rybski et al. (2010):** Demonstrated HOG effectiveness across varied poses.  
- **Ghaffari et al. (2020):** Explored FPGA-based HOG optimization for large-scale computation.  

These works highlight the strong generalizability and robustness of HOG across image types and environments.  
(:contentReference[oaicite:4]{index=4})

---

# 🔬 Methods

## 📁 Dataset
The **LHI Animal Faces dataset** contains:

- **20 classes** (19 animal species + humans)
- Each image containing only the subject’s **face**
- Visual examples appear in Figure 1 of the PDF (:contentReference[oaicite:5]{index=5})

Class imbalance is shown in Figure 2.  
Dog and Cat images appear most frequently, while several classes contain fewer samples.

---

## 🛠 Preprocessing
Each image undergoes three preprocessing steps (Figure 3):

1. **Grayscale conversion**  
2. **Histogram equalization** for contrast enhancement  
3. **Gaussian noise injection** to reduce overfitting and improve robustness  

(:contentReference[oaicite:6]{index=6})

---

## ⚙️ Feature Extraction Methods

### 1. **Sobel Operator**
- Computes horizontal + vertical gradients  
- Produces edge magnitude and direction maps  
- Features: gradient statistics and pixel intensity gradients  
(:contentReference[oaicite:7]{index=7})

---

### 2. **Laplacian Operator**
- Second-order derivative emphasizing rapid intensity changes  
- Captures edges uniformly across directions  
- Features include mean, SD, and max edge intensity in a **4×4 grid**  
(:contentReference[oaicite:8]{index=8})

---

### 3. **Histogram of Oriented Gradients (HOG)**
- Computes gradient histograms within **8×8 pixel cells**  
- Normalizes across spatial blocks  
- Produces detailed shape + contour descriptors  
- Most effective method in this project  
(:contentReference[oaicite:9]{index=9})

---

## 🤖 Model Training & Evaluation
- Training/testing split: **80% / 20%**
- Classifier: **Support Vector Classifier (SVC)**
- Each feature extraction technique was evaluated separately.

After HOG outperformed the others, multiple ML models were tested (Table 1):

| Model | Accuracy |
|-------|----------|
| **Support Vector Machine** | **0.77** |
| Logistic Regression | 0.76 |
| K-Nearest Neighbors | 0.39 |
| Random Forest | 0.45 |
| SGD Classifier | 0.62 |

(:contentReference[oaicite:10]{index=10})  
SVC was selected as the final model.

Hyperparameters (from GridSearchCV):

| Parameter | Value |
|-----------|--------|
| C | 0.1 |
| degree | 2 |
| gamma | scale |
| kernel | linear |

(:contentReference[oaicite:11]{index=11})

---

# 📊 Results

### ⭐ HOG Produced the Best Results
Final performance metrics (Table 3):

- **Precision:** 0.774  
- **Recall:** 0.746  
- **F1-score:** 0.747  
- **Accuracy:** 0.746  

HOG features consistently outperform Sobel and Laplacian across all classes.

Class-specific examples (Table 3):

- **Human:** F1 = 0.974  
- **Tiger:** F1 = 0.894  
- **Dog:** F1 = 0.728  
- **Mouse:** F1 = 0.474  

(:contentReference[oaicite:12]{index=12})

---

# 🌐 Model Deployment
After training:

- Preprocessing + feature extraction  
- SVC prediction pipeline  
- Flask backend  
- Simple HTML frontend (Figure 4)

Users can upload an image and receive the **top 5 predicted classes**.  
(:contentReference[oaicite:13]{index=13})

---

# 🏁 Conclusion
Traditional feature extraction techniques remain effective alternatives to deep learning when:

- Datasets are small  
- Interpretability is required  
- Computational resources are limited  

HOG combined with an SVC model offers a high-performing, interpretable image classification pipeline for animal faces.  
(:contentReference[oaicite:14]{index=14})

---

# ⚠️ Limitations
- Model trained **only on facial images**  
- Does not generalize to **full-body images**  
- Poor performance in classes with low sample size  
- No pose-invariance or background handling  
(:contentReference[oaicite:15]{index=15})

---

# 🔮 Future Work
Recommended enhancements (Section 5.2):

- Integrate **YOLO** or **Haar cascade** face detection to process full-body images  
- Add **color moments** to complement gradient-based features  
- Expand dataset to include more poses, lighting conditions, and backgrounds  
(:contentReference[oaicite:16]{index=16})

---

# 📎 Appendix – Classification Metrics
Full metrics for all three feature extraction methods are included:

- **Table 4:** Sobel metrics  
- **Table 5:** Laplacian metrics  
- **Table 6:** HOG metrics  

These tables provide per-class precision, recall, F1 score, and support.  
(:contentReference[oaicite:17]{index=17})

---

# 📚 References
See full reference list on page 10 of the project PDF.  
(:contentReference[oaicite:18]{index=18})

---

# 📫 Contact
**George Tzimas**  
📧 georgetz95@gmail.com  
🔗 GitHub: https://github.com/georgetz95  
🔗 LinkedIn: https://www.linkedin.com/in/georgetz95
