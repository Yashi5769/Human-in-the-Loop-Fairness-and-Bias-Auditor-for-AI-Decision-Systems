# Human-in-the-Loop Fairness and Bias Auditor for AI Decision Systems

## Overview

The **Human-in-the-Loop Fairness and Bias Auditor** is an end-to-end AI governance dashboard built using **Python**, **Streamlit**, and **Machine Learning**.

The system helps organizations monitor, explain, detect, and mitigate bias in AI decision-making systems while keeping humans involved in the review process.

This project goes beyond traditional machine learning by focusing not only on **accuracy**, but also on:

- Fairness
- Transparency
- Explainability
- Bias Detection
- Bias Mitigation
- Human Oversight
- Model Governance
- Continuous Monitoring

The dashboard uses the **Adult Income Dataset** to predict whether a person earns more than \$50K annually while auditing the fairness of those predictions across demographic groups.

---

# Features

## 1. Model Training & Prediction

- Trains a baseline AI model on the Adult Income dataset
- Predicts whether income is:
  - `<=50K`
  - `>50K`

Supports:
- Logistic Regression
- Random Forest
- XGBoost

---

## 2. Fairness Evaluation

Computes fairness metrics including:

### Demographic Parity Difference (DP Difference)

Measures whether positive outcomes are equally distributed across groups.

### Disparate Impact Ratio (DI Ratio)

Measures whether one demographic group receives favorable outcomes significantly more often than another.

---

## 3. Multi-Model Comparison

Compares multiple ML models based on:

- Accuracy
- Fairness
- Final weighted score

Automatically recommends the best model using configurable fairness vs accuracy weights.

---

## 4. Bias Heatmaps

Visualizes fairness across demographic groups using heatmaps.

Includes:

- Gender fairness heatmaps
- Intersectional fairness heatmaps (Gender × Race)

---

## 5. Intersectional Fairness Analysis

Detects hidden discrimination across combinations of sensitive attributes such as:

- Female + Asian
- Male + Black
- Female + White

---

## 6. Continuous Fairness Monitoring

Tracks fairness metrics over time and detects:

- Bias drift
- Distribution changes
- Performance degradation

---

## 7. Live Fairness Alert System

Generates alerts when:

- DP Difference exceeds threshold
- DI Ratio becomes unsafe
- Counterfactual bias is detected

---

## 8. Auto-Retraining with Version Control

Automatically:

- Retrains models when fairness deteriorates
- Applies mitigation strategies
- Saves improved versions
- Rolls back unfair versions

---

## 9. Bias Mitigation using Group Reweighting

Identifies the most disadvantaged demographic group and increases its training importance using:

- Sample reweighting
- Fairness-aware retraining

---

## 10. SHAP Explainability

Provides:

- Global feature importance
- Per-instance explanations
- Waterfall plots
- Feature contribution breakdowns

Helps explain:

> Why the model made a specific prediction.

---

## 11. Counterfactual Fairness Testing

Tests whether predictions change unfairly when only sensitive attributes are modified.

Example:

- Same profile
- Only gender changes
- Prediction changes → bias detected

---

## 12. Human-in-the-Loop Review

Allows humans to:

- Review predictions
- Override incorrect decisions
- Log corrections
- Maintain accountability

---

## 13. What-if Simulator

Interactive simulator to test:

- Age changes
- Work hour changes
- Education changes
- Gender changes

Shows how predictions and fairness respond dynamically.

---

# Tech Stack

## Frontend

- Streamlit

## Backend

- Python

## Machine Learning

- Scikit-learn
- XGBoost

## Explainability

- SHAP

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn

# Project Structure

```bash
fairness_auditor_project/
│
├── app/
│   └── dashboard.py
│
├── data/
│   ├── adult/
│   │   └── adult.data
│   ├── fairness_log.csv
│   └── human_feedback.csv
│
├── models/
│   ├── saved_models/
│   └── model_log.csv
│
├── src/
│   ├── auto_retrain.py
│   ├── baseline_model.py
│   ├── bias_drift_detection.py
│   ├── counterfactual_analysis.py
│   ├── data_preprocessing.py
│   ├── explainability.py
│   ├── fairness_metrics.py
│   ├── fairness_monitoring.py
│   ├── group_bias_mitigation.py
│   ├── human_review.py
│   ├── model_comparison.py
│   └── model_versioning.py
│
├── requirements.txt
└── README.md
Installation
1. Clone Repository
git clone https://github.com/yourusername/fairness-auditor.git

cd fairness-auditor

