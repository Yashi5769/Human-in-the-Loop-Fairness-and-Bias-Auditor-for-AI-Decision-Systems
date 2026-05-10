import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.data_preprocessing import load_data, preprocess_data
from src.baseline_model import train_baseline_model
from src.fairness_metrics import (
    demographic_parity_difference,
    disparate_impact_ratio,
    group_outcome_rates
)

from src.explainability import shap_analysis
from src.counterfactual_analysis import counterfactual_bias_rate
from src.fairness_alerts import fairness_alert_system


# --------------------------------
# PHASE 1 – DATA PREPARATION
# --------------------------------

df = load_data("data/adult.data")

df_processed, sensitive = preprocess_data(df)


# --------------------------------
# PHASE 2 – BASELINE MODEL
# --------------------------------

model, X_train, X_test, y_train, y_test, predictions, accuracy = train_baseline_model(df_processed)

print("\nModel Accuracy:", accuracy)


# Confusion Matrix

cm = confusion_matrix(y_test, predictions)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["≤50K",">50K"],
    yticklabels=["≤50K",">50K"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# --------------------------------
# PHASE 3 – FAIRNESS METRICS
# --------------------------------

sex_feature = X_test["sex_Male"]

dp = demographic_parity_difference(predictions, sex_feature)

di = disparate_impact_ratio(predictions, sex_feature)

print("\nDemographic Parity Difference:", dp)
print("Disparate Impact Ratio:", di)


rates = group_outcome_rates(predictions, sex_feature)

rates.plot(kind="bar")

plt.title("Outcome Rate by Gender")

plt.ylabel("Positive Prediction Rate")

plt.show()


# --------------------------------
# PHASE 4 – EXPLAINABILITY
# --------------------------------

X_sample = X_test.sample(min(100, len(X_test))).astype(float)

shap_values = shap_analysis(model, X_sample)


# --------------------------------
# COUNTERFACTUAL FAIRNESS
# --------------------------------

counterfactual_rate = counterfactual_bias_rate(model, X_test)

print("\nCounterfactual Bias Rate:", counterfactual_rate)


# --------------------------------
# FAIRNESS ALERT SYSTEM
# --------------------------------

alerts = fairness_alert_system(dp, di, counterfactual_rate)