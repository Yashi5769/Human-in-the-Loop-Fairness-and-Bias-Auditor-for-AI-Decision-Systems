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
```
# Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/fairness-auditor.git

cd fairness-auditor
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Libraries

```txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
joblib
```

---

# Running the Dashboard

```bash
streamlit run app/dashboard.py
```

The app will open at:

```txt
http://localhost:8501
```

---

# Dataset

## Adult Income Dataset

The dataset predicts whether an individual's income exceeds \$50K/year based on demographic attributes.

### Features Include:

- Age
- Education
- Occupation
- Race
- Gender
- Hours per week
- Capital gain/loss

---

# Dashboard Sections

---

## Model Overview

Displays:

- Dataset size
- Feature count
- Accuracy
- Sensitive attribute being monitored

---

## Prediction Distribution

Shows:

- Number of low-income predictions
- Number of high-income predictions

---

## Fairness Evaluation

Displays:

- Demographic Parity Difference
- Disparate Impact Ratio
- Fairness status

---

## Multi-Model Comparison

Compares:

- Logistic Regression
- Random Forest
- XGBoost

Across:

- Accuracy
- Fairness
- Final weighted score

---

## Continuous Fairness Monitoring

Tracks fairness over time and detects:

- Drift
- Sudden fairness degradation

---

## Auto-Retrain System

Automatically:

- Retrains unfair models
- Saves improved versions
- Rolls back degraded models

---

## Bias Heatmaps

Visual fairness comparison across:

- Gender
- Race
- Intersectional groups

---

## Bias Mitigation

Uses:

- Group reweighting
- Fairness-aware retraining

To reduce discrimination.

---

## SHAP Explainability

Explains:

- Which features influence predictions
- Why individual predictions happen

---

## Counterfactual Fairness

Tests whether changing only sensitive attributes changes predictions unfairly.

---

## Human Review

Allows:

- Manual correction
- Human feedback logging
- AI oversight

---

## What-if Simulator

Interactive testing tool for:

- Scenario analysis
- Fairness stress testing
- Feature impact exploration

---

# Fairness Metrics Explained

## Demographic Parity Difference

Measures:

> Difference in positive prediction rates between groups.

Ideal value:

```txt
0
```

---

## Disparate Impact Ratio

Measures:

> Ratio of favorable outcomes between groups.

Ideal value:

```txt
1
```

---

# Explainability

The project uses **SHAP (SHapley Additive exPlanations)** for:

- Global interpretability
- Local interpretability
- Feature attribution

---

# Bias Mitigation Strategy

## Group Reweighting

The system:

1. Detects the worst affected demographic group
2. Assigns higher training weights
3. Retrains the model
4. Re-evaluates fairness

---

# Human-in-the-Loop Concept

Instead of fully autonomous AI:

- Humans remain involved
- Decisions can be reviewed
- Bias can be corrected manually

This improves:

- Accountability
- Trust
- Ethical AI deployment

---

# Future Improvements

Possible future enhancements:

- Real-time deployment monitoring
- Advanced fairness metrics
- LLM-powered explanations
- Federated fairness auditing
- API integration
- Role-based authentication
- PDF fairness reports
- Cloud deployment

---

# Use Cases

This system can be applied in:

- Hiring systems
- Loan approval systems
- Healthcare AI
- Insurance AI
- University admissions
- Criminal justice systems

---

# Learning Outcomes

This project demonstrates:

- Machine Learning Engineering
- Responsible AI
- AI Fairness
- Explainable AI (XAI)
- Human-in-the-Loop Systems
- AI Governance
- Ethical AI Deployment
- MLOps Concepts

---

# Author

## Yashi

B.Tech CSE (AI/ML Specialization)

Focused on:

- Responsible AI
- Fairness in Machine Learning
- Explainable AI
- Human-Centered AI Systems

---

# License

This project is licensed under the MIT License.

---

# Final Summary

> This project transforms a traditional machine learning pipeline into a fully monitored, explainable, fairness-aware, and human-supervised AI governance system capable of detecting, mitigating, and continuously monitoring bias in real-world AI decision systems.
