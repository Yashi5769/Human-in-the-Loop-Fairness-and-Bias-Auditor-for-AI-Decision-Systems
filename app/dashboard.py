import sys
import os
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.data_preprocessing import load_data, preprocess_data
from src.baseline_model import train_baseline_model
from src.fairness_metrics import demographic_parity_difference, disparate_impact_ratio
from src.explainability import (
    compute_shap_values,
    compute_shap_explainer,
    plot_shap_summary,
    compute_shap_explainer,
    get_shap_values,
    plot_local_explanation,
    plot_feature_contribution_bar
)

from src.counterfactual_analysis import counterfactual_test
from src.bias_drift_detection import detect_bias_drift
from src.human_review import log_human_feedback
from src.model_comparison import train_models_and_compare, recommend_best_model
from src.fairness_monitoring import log_fairness, check_drift
from src.auto_retrain import auto_retrain
from src.model_versioning import save_model, load_latest_model, rollback_model
from src.group_bias_mitigation import (
    identify_worst_group,
    apply_reweighting,
    retrain_with_weights
)

def enforce_numeric(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.fillna(0)
# ------------------------------------------------
# CLEAN SETUP
# ------------------------------------------------

plt.close('all')

def clean_dataframe(df):

    df = df.copy()

    for col in df.columns:

        # bool → int
        if df[col].dtype == bool:

            df[col] = df[col].astype(int)

        # object → numeric IF POSSIBLE
        elif df[col].dtype == object:

            converted = pd.to_numeric(
                df[col],
                errors="ignore"
            )

            df[col] = converted

    return df

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(page_title="AI Fairness Dashboard", layout="wide")
st.title("Human-in-the-Loop Fairness and Bias Auditor for AI Decision Systems Dashboard")

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #161B22;
    border: 1px solid #30363D;
    padding: 15px;
    border-radius: 12px;
}

/* Buttons */
.stButton button {
    background-color: #238636;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.5rem 1rem;
}

/* Headers */
h1, h2, h3 {
    color: #F0F6FC;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161B22;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")

section = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Model Comparison",
        "Fairness",
        "Explainability",
        "Bias Mitigation",
        "What-if Simulator"
    ]
)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

data_path = os.path.join(BASE_DIR, "data", "adult", "adult.data")

# fallback if not found
if not os.path.exists(data_path):
    data_path = "data/adult/adult.data"

if not os.path.exists(data_path):
    st.error("Dataset not found. Please check data/adult/adult.data")
    st.stop()

@st.cache_data
def load_dataset(path):
    df = load_data(path)
    return df, preprocess_data(df)

df_raw, (df_processed, sensitive) = load_dataset(data_path)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

@st.cache_resource
def load_model(_df_processed):
    return train_baseline_model(_df_processed)

# initialize ALL state variables together
required_keys = [
    "model",
    "X_test",
    "y_test",
    "predictions",
    "accuracy"
]

missing_keys = [
    key for key in required_keys
    if key not in st.session_state
]

if len(missing_keys) > 0:

    model, X_test, y_test, predictions, accuracy = load_model(df_processed)

    st.session_state["model"] = model
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["predictions"] = predictions
    st.session_state["accuracy"] = accuracy

# retrieve safely
model = st.session_state["model"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]
predictions = st.session_state["predictions"]
accuracy = st.session_state["accuracy"]

if len(X_test) == 0:
    st.error("No test data available")
    st.stop()

if "sex_Male" not in X_test.columns:
    st.error("Missing 'sex_Male' feature")
    st.stop()

sex_feature = X_test["sex_Male"]

dp = demographic_parity_difference(predictions, sex_feature)
di = disparate_impact_ratio(predictions, sex_feature)

# ------------------------------------------------
# TOP METRICS
# ------------------------------------------------

sns.set_style("whitegrid")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Accuracy", round(accuracy, 3))
k2.metric("DP Difference", round(dp, 3))
k3.metric("DI Ratio", round(di, 3))
k4.metric("Samples", len(df_raw))

# ------------------------------------------------
# PREDICTION DISTRIBUTION
# ------------------------------------------------

st.subheader("Prediction Distribution")

dist = (
    pd.Series(predictions)
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

dist_df = pd.DataFrame({
    "Income": ["≤50K", ">50K"],
    "Count": dist.values
})

fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(
    dist_df["Income"],
    dist_df["Count"]
)

ax.set_title("Prediction Distribution")
ax.set_xlabel("Income")
ax.set_ylabel("Count")

for i, v in enumerate(dist_df["Count"]):
    ax.text(i, v + 5, str(v), ha="center")

st.pyplot(fig)

plt.close(fig)


# ------------------------------------------------
# MODEL OVERVIEW
# ------------------------------------------------

st.header("Model Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Samples", f"{len(df_raw):,}")
col2.metric("Features", len(df_processed.columns)-1)
col3.metric("Accuracy", round(accuracy, 3))
col4.metric("Sensitive Attribute", "Gender")

st.divider()

# ------------------------------------------------
# MULTI-MODEL COMPARISON
# ------------------------------------------------

# ------------------------------------------------
# MULTI-MODEL COMPARISON
# ------------------------------------------------

@st.cache_data
def cached_model_results(df):
    return train_models_and_compare(df)

results = cached_model_results(df_processed)

results_df = pd.DataFrame(results)

# -----------------------------
# FORCE NUMERIC TYPES
# -----------------------------

numeric_cols = [
    "Accuracy",
    "DP Difference",
    "DI Ratio"
]

for col in numeric_cols:
    results_df[col] = pd.to_numeric(
        results_df[col],
        errors="coerce"
    )

results_df = results_df.fillna(0)

# -----------------------------
# DISPLAY TABLE
# -----------------------------

st.header("Multi-Model Comparison")

st.dataframe(
    results_df.reset_index(drop=True),
    use_container_width=True
)

# ------------------------------------------------
# MODEL RECOMMENDATION
# ------------------------------------------------

st.header("Automatic Model Recommendation")

alpha = st.slider(
    "Accuracy Weight",
    0.0,
    1.0,
    0.5
)

beta = 1 - alpha

df_scores, best_model = recommend_best_model(
    results,
    alpha,
    beta
)

# -----------------------------
# CLEAN NUMERIC TYPES
# -----------------------------

score_cols = [
    "Accuracy",
    "DP Difference",
    "DI Ratio",
    "Fairness Score",
    "Final Score"
]

for col in score_cols:

    df_scores[col] = pd.to_numeric(
        df_scores[col],
        errors="coerce"
    )

df_scores = df_scores.fillna(0)

# -----------------------------
# SHOW TABLE
# -----------------------------

st.dataframe(
    df_scores.reset_index(drop=True),
    use_container_width=True
)

# -----------------------------
# BEST MODEL CARD
# -----------------------------

st.success(
    f"""
Best Model: {best_model['Model']}

Accuracy: {best_model['Accuracy']:.3f}

DP Difference: {best_model['DP Difference']:.3f}

Final Score: {best_model['Final Score']:.3f}
"""
)

# ------------------------------------------------
# INTERACTIVE PLOTLY CHART
# ------------------------------------------------

st.subheader("Model Final Scores")

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    df_scores["Model"],
    df_scores["Final Score"]
)

ax.set_title("Final Model Ranking")
ax.set_xlabel("Model")
ax.set_ylabel("Final Score")

for i, v in enumerate(df_scores["Final Score"]):
    ax.text(i, v + 0.01, round(v, 3), ha="center")

st.pyplot(fig)

plt.close(fig)

st.divider()

# ------------------------------------------------
# FAIRNESS METRICS
# ------------------------------------------------

st.header("Fairness Evaluation")

col1, col2 = st.columns(2)
col1.metric("DP Difference", round(dp, 3))
col2.metric("DI Ratio", round(di, 3))

if abs(dp) > 0.1:
    st.error("⚠️ Bias Detected")
else:
    st.success("Fair Model")

st.divider()

dp = demographic_parity_difference(predictions, sex_feature)
di = disparate_impact_ratio(predictions, sex_feature)

# ------------------------------------------------
# TOP METRICS
# ------------------------------------------------

sns.set_style("whitegrid")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Accuracy", round(accuracy, 3))
k2.metric("DP Difference", round(dp, 3))
k3.metric("DI Ratio", round(di, 3))
k4.metric("Samples", len(df_raw))

# ------------------------------------------------
# CONTINUOUS FAIRNESS MONITORING
# ------------------------------------------------

st.header("Continuous Fairness Monitoring")

# log current fairness
if "logged" not in st.session_state:
    log_fairness(dp, di)
    st.session_state["logged"] = True

# check drift
drift_flag, drift_value = check_drift()

st.metric("Fairness Drift (DP change)", round(drift_value, 4))

if drift_flag:
    st.error("🚨 Fairness Drift Detected")

    retrain_trigger = st.button("Trigger Auto-Retraining")

    if retrain_trigger:

        st.info("Retraining model with fairness mitigation...")

        new_model, X_new, y_new, new_preds = auto_retrain(df_processed)

        new_dp = demographic_parity_difference(new_preds, X_new["sex_Male"])

        st.success(f"Retrained Model DP: {round(new_dp,3)}")

else:
    st.success("No fairness drift detected")
# ------------------------------------------------
# AUTO RETRAIN WITH VERSIONING + ROLLBACK
# ------------------------------------------------

st.subheader("Auto-Retrain with Version Control")

if st.button("Trigger Retraining"):

    st.info("Retraining model...")

    # ------------------------------------------------
    # BUILD CLEAN MITIGATION DATASET
    # ------------------------------------------------

    mitigation_df = X_test.copy().reset_index(drop=True)

    # add true labels (recommended)
    mitigation_df["income"] = (
        pd.Series(y_test)
        .reset_index(drop=True)
    )

    # gender labels
    mitigation_df["gender"] = (
        mitigation_df["sex_Male"]
        .map({
            0: "Female",
            1: "Male"
        })
    )

    # ------------------------------------------------
    # IDENTIFY WORST GROUP
    # ------------------------------------------------

    worst_group, group_rates = identify_worst_group(
        mitigation_df.rename(columns={
            "income": "prediction"
        }),
        ["gender"]
    )

    # ------------------------------------------------
    # APPLY REWEIGHTING
    # ------------------------------------------------

    mitigation_df = apply_reweighting(
        mitigation_df,
        worst_group,
        ["gender"]
    )

    # ------------------------------------------------
    # RETRAIN MODEL
    # ------------------------------------------------

    new_model, X_mit_test, y_mit_test, mit_preds = (
        retrain_with_weights(
            mitigation_df,
            target_col="income"
        )
    )

    # ------------------------------------------------
    # FAIRNESS CHECK
    # ------------------------------------------------

    new_dp = demographic_parity_difference(
        mit_preds,
        X_mit_test["sex_Male"]
    )

    st.write(
        f"New Model DP: {round(new_dp, 3)}"
    )

    st.write(
        f"Current Model DP: {round(dp, 3)}"
    )

    # ------------------------------------------------
    # FAIRNESS TOLERANCE
    # ------------------------------------------------

    FAIRNESS_TOLERANCE = 0.01

    # ------------------------------------------------
    # DEPLOYMENT DECISION
    # ------------------------------------------------

    if abs(new_dp) <= abs(dp) + FAIRNESS_TOLERANCE:

        save_model(new_model, new_dp)

        st.session_state["model"] = new_model

        st.success(
            "✅ Model improved & saved"
        )

        st.info(f"""
Deployment Decision

Old DP: {round(dp, 3)}
New DP: {round(new_dp, 3)}

Action: DEPLOYED
""")

    else:

        st.error(
            "🚨 Fairness worsened — rolling back"
        )

        rollback = rollback_model()

        if rollback:

            st.session_state["model"] = rollback

            st.success(
                "Rolled back to previous model"
            )

            st.info(f"""
Deployment Decision

Old DP: {round(dp, 3)}
New DP: {round(new_dp, 3)}

Action: ROLLED BACK
""")

        else:

            st.warning(
                "No previous model available"
            )

# ------------------------------------------------
# MODEL VERSION HISTORY
# ------------------------------------------------

st.subheader("Model Version History")

log_path = "models/model_log.csv"

if os.path.exists(log_path):

    hist = pd.read_csv(log_path)

    st.dataframe(hist)

    fig, ax = plt.subplots()

    ax.plot(hist["dp"], marker='o')

    ax.set_title("Fairness Across Model Versions")
    ax.set_ylabel("DP Difference")

    st.pyplot(fig)
    plt.close(fig)
# ------------------------------------------------
# LIVE FAIRNESS ALERT SYSTEM
# ------------------------------------------------

st.header("Live Fairness Alert System")

# configurable thresholds
dp_threshold = st.slider("DP Threshold", 0.05, 0.5, 0.1)
di_lower = 0.8
di_upper = 1.25

alerts = []

# check DP
if abs(dp) > dp_threshold:
    alerts.append(f"High Demographic Parity Difference: {round(dp,3)}")

# check DI
if di < di_lower or di > di_upper:
    alerts.append(f"Disparate Impact out of range: {round(di,3)}")

# check counterfactual fairness (sample-based quick test)
if len(X_test) > 0:
    sample_cf = X_test.iloc[0]
    orig_pred_cf, counter_pred_cf = counterfactual_test(model, sample_cf, "sex_Male")

if orig_pred_cf != counter_pred_cf:
    alerts.append("Counterfactual bias detected")

# display alerts
if alerts:
    st.error("🚨 Fairness Issues Detected")
    
    for alert in alerts:
        st.warning(alert)

else:
    st.success("✅ System is Fair within defined thresholds")

# ------------------------------------------------
# BIAS HEATMAP (SAFE VERSION)
# ------------------------------------------------

st.header("Bias Heatmap (Group Fairness Overview)")

# ------------------------------------------------
# BUILD DATAFRAME
# ------------------------------------------------

heatmap_df = pd.DataFrame({
    "prediction": pd.Series(predictions).astype(float),
    "gender": pd.Series(sex_feature).astype(int)
})

# map labels
heatmap_df["gender"] = heatmap_df["gender"].map({
    0: "Female",
    1: "Male"
})

# remove invalid rows
heatmap_df = heatmap_df.dropna()

# ------------------------------------------------
# EMPTY CHECK
# ------------------------------------------------

if heatmap_df.empty:

    st.warning("Not enough data to compute heatmap")

else:

    # ------------------------------------------------
    # GROUP POSITIVE RATES
    # ------------------------------------------------

    pivot = pd.crosstab(
        heatmap_df["gender"],
        heatmap_df["prediction"],
        normalize="index"
    )

    # ensure numeric
    pivot = pivot.astype(float)

    # ensure both prediction columns exist
    pivot = pivot.reindex(
        columns=[0.0, 1.0],
        fill_value=0
    )

    # rename prediction columns
    pivot.columns = ["≤50K", ">50K"]

    # ------------------------------------------------
    # PLOT
    # ------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title("Positive Outcome Rate by Gender")

    st.pyplot(fig)

    plt.close(fig)

# ------------------------------------------------
# INTERSECTIONAL FAIRNESS DASHBOARD (Gender × Race)
# ------------------------------------------------

st.header("Intersectional Fairness (Gender × Race)")

# ---- SAFE DATA BUILD ----
fair_df = pd.DataFrame({
    "prediction": pd.Series(predictions).astype(float),
    "gender": pd.Series(sex_feature).astype(int)
})

# Try to get race feature safely
race_cols = [col for col in X_test.columns if "race_" in col]

if len(race_cols) == 0:
    st.warning("Race features not found in dataset")
else:
    # convert one-hot race → label
    race_data = X_test[race_cols].copy()

    # ensure numeric
    race_data = race_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    fair_df["race"] = race_data.idxmax(axis=1).str.replace("race_", "")

    # map gender
    fair_df["gender"] = fair_df["gender"].map({0: "Female", 1: "Male"})

    # drop bad rows
    fair_df = fair_df.dropna()

    if fair_df.empty:
        st.warning("Not enough data for intersectional fairness")
    else:

        # ---- CREATE INTERSECTION GROUP ----
        fair_df["group"] = fair_df["gender"] + " | " + fair_df["race"]

        # ---- GROUP METRICS ----
        group_rates = fair_df.groupby("group")["prediction"].mean().sort_values()

        # ------------------------------------------------
        # 📊 BAR CHART
        # ------------------------------------------------
        st.subheader("Positive Outcome Rate by Group")

        fig, ax = plt.subplots(figsize=(10, 4))

        group_rates.plot(kind="bar", ax=ax)

        ax.set_ylabel("Positive Rate (>50K)")
        ax.set_xlabel("Group")
        ax.set_title("Intersectional Fairness")

        st.pyplot(fig)
        plt.close(fig)

        # ------------------------------------------------
        # 🔥 HEATMAP (Gender vs Race)
        # ------------------------------------------------
        st.subheader("Heatmap (Gender vs Race)")

        pivot = fair_df.pivot_table(
    values="prediction",
    index="gender",
    columns="race",
    aggfunc="mean"
)

        pivot = pivot.fillna(0)

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.heatmap(
    pivot.astype(float),
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    ax=ax
)

        ax.set_title("Fairness Heatmap")

        st.pyplot(fig)

        plt.close(fig)

        # ------------------------------------------------
        # 🚨 FAIRNESS GAP DETECTION
        # ------------------------------------------------
        st.subheader("Fairness Gap Analysis")

        max_rate = group_rates.max()
        min_rate = group_rates.min()
        gap = max_rate - min_rate

        st.metric("Max Gap Across Groups", round(gap, 3))

        if gap > 0.2:
            st.error("🚨 High disparity across intersectional groups")
        elif gap > 0.1:
            st.warning("⚠️ Moderate disparity detected")
        else:
            st.success("✅ Fair distribution across groups")

# ------------------------------------------------
# GROUP BIAS MITIGATION
# ------------------------------------------------


st.header("Bias Mitigation (Group Reweighting)")

# ------------------------------------------------
# PREP DATA
# ------------------------------------------------

# use ONLY test data to match prediction length
mitigation_df = X_test.copy().reset_index(drop=True)

# add model predictions safely
mitigation_df["prediction"] = pd.Series(predictions).reset_index(drop=True)

# ------------------------------------------------
# GENDER LABELS
# ------------------------------------------------

mitigation_df["gender"] = mitigation_df["sex_Male"].map({
    0: "Female",
    1: "Male"
})

# ------------------------------------------------
# RACE LABELS
# ------------------------------------------------

race_cols = [c for c in mitigation_df.columns if "race_" in c]

if len(race_cols) > 0:

    mitigation_df["race"] = (
        mitigation_df[race_cols]
        .astype(float)
        .idxmax(axis=1)
        .str.replace("race_", "", regex=False)
    )

else:
    mitigation_df["race"] = "Unknown"

# ------------------------------------------------
# CLEAN DATA
# ------------------------------------------------

mitigation_df = mitigation_df.fillna(0)

# ------------------------------------------------
# IDENTIFY WORST GROUP
# ------------------------------------------------

worst_group, group_rates = identify_worst_group(
    mitigation_df,
    ["gender", "race"]
)

st.subheader("Worst Affected Group")

st.error(str(worst_group))

# ------------------------------------------------
# DISPLAY GROUP RATES
# ------------------------------------------------

group_rates_df = (
    group_rates.reset_index()
    .rename(columns={"prediction": "Positive Outcome Rate"})
)

st.dataframe(group_rates_df)

# ------------------------------------------------
# APPLY REWEIGHTING
# ------------------------------------------------

mitigation_df = apply_reweighting(
    mitigation_df,
    worst_group,
    ["gender", "race"]
)

# ------------------------------------------------
# RETRAIN MODEL
# ------------------------------------------------

mit_model, X_mit_test, y_mit_test, mit_preds = retrain_with_weights(
    mitigation_df,
    target_col="prediction"
)

# ------------------------------------------------
# FAIRNESS AFTER MITIGATION
# ------------------------------------------------

if "sex_Male" not in X_mit_test.columns:

    st.error("sex_Male column missing after mitigation")

else:

    mit_dp = demographic_parity_difference(
        mit_preds,
        X_mit_test["sex_Male"]
    )

    # ------------------------------------------------
    # RESULTS
    # ------------------------------------------------

    col1, col2 = st.columns(2)

    col1.metric(
        "Before Mitigation DP",
        round(dp, 3)
    )

    col2.metric(
        "After Mitigation DP",
        round(mit_dp, 3)
    )

    # ------------------------------------------------
    # STATUS MESSAGE
    # ------------------------------------------------

    if abs(mit_dp) < abs(dp):

        st.success(
            "✅ Fairness Improved After Reweighting"
        )

    elif abs(mit_dp) == abs(dp):

        st.info(
            "ℹ️ Fairness unchanged"
        )

    else:

        st.warning(
            "⚠️ Fairness worsened — tune reweighting"
        )

    # ------------------------------------------------
    # VISUAL COMPARISON
    # ------------------------------------------------

    comparison_df = pd.DataFrame({
        "Stage": ["Before", "After"],
        "DP Difference": [dp, mit_dp]
    })

    fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(
    comparison_df["Stage"],
    comparison_df["DP Difference"]
)

ax.set_title(
    "Fairness Improvement After Mitigation"
)

ax.set_ylabel("DP Difference")

for i, v in enumerate(comparison_df["DP Difference"]):
    ax.text(i, v + 0.01, round(v, 3), ha="center")

st.pyplot(fig)

plt.close(fig)
# ------------------------------------------------
# SHAP EXPLAINABILITY (INTERACTIVE)
# ------------------------------------------------

st.header("SHAP Explainability")

@st.cache_resource
def load_explainer(_model, X):
    return compute_shap_explainer(_model, X)

if st.checkbox("Enable Explainability"):

    X_sample = X_test.sample(min(200, len(X_test)))
    X_sample = enforce_numeric(X_sample)
    X_sample = X_sample.apply(pd.to_numeric, errors='coerce').fillna(0)

    explainer = load_explainer(model, X_sample)
    with st.spinner("Computing SHAP values..."):
         shap_values = get_shap_values(explainer, X_sample)

    # -------------------------
    # GLOBAL VIEW
    # -------------------------
    st.subheader("Global Feature Importance")

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig)

    st.divider()

    # -------------------------
    # LOCAL EXPLANATION
    # -------------------------
    st.subheader("Per-Instance Explanation")

    index = st.slider("Select Data Point", 0, len(X_sample)-1)

    st.write("Selected Data Point Features:")
    st.dataframe(X_sample.iloc[index:index+1])

    # Waterfall Plot
    st.subheader("Waterfall Explanation")

    fig_local = plot_local_explanation(shap_values, X_sample, index)
    st.pyplot(fig_local)

    # Bar Contribution Plot
    st.subheader("Feature Contribution Breakdown")

    fig_bar = plot_feature_contribution_bar(shap_values, X_sample, index)
    st.pyplot(fig_bar)

# ------------------------------------------------
# COUNTERFACTUAL FAIRNESS
# ------------------------------------------------

st.header("Counterfactual Fairness")

sample_index = st.slider("Select Case", 0, len(X_test)-1)
sample = X_test.iloc[sample_index]

orig_pred, counter_pred = counterfactual_test(model, sample, "sex_Male")

st.write("Original:", orig_pred)
st.write("Counterfactual:", counter_pred)

if orig_pred != counter_pred:
    st.error("Bias detected")
else:
    st.success("Stable prediction")

st.divider()

# ------------------------------------------------
# DRIFT MONITORING
# ------------------------------------------------

st.header("Bias Drift Monitoring")

drift_flag, drift_value = detect_bias_drift(0.10, dp)

if drift_flag:
    st.error(f"Drift detected: {round(drift_value,3)}")
else:
    st.success("No drift")

st.divider()

# ------------------------------------------------
# FAIRNESS HISTORY
# ------------------------------------------------

st.subheader("Fairness Over Time")

log_path = "data/fairness_log.csv"

if not os.path.exists(log_path):

    st.warning("No fairness history available yet.")

else:

    hist_df = pd.read_csv(log_path)

    # ----------------------------------------
    # DEBUG VIEW
    # ----------------------------------------

    st.write("Raw Fairness Log")
    st.dataframe(hist_df.head())

    # ----------------------------------------
    # REQUIRED COLUMNS CHECK
    # ----------------------------------------

    required_cols = ["dp", "di"]

    missing_cols = [
        col for col in required_cols
        if col not in hist_df.columns
    ]

    if len(missing_cols) > 0:

        st.error(
            f"Missing columns in fairness_log.csv: {missing_cols}"
        )

    else:

        # ----------------------------------------
        # FORCE NUMERIC
        # ----------------------------------------

        hist_df["dp"] = pd.to_numeric(
            hist_df["dp"],
            errors="coerce"
        )

        hist_df["di"] = pd.to_numeric(
            hist_df["di"],
            errors="coerce"
        )

        # remove bad rows
        hist_df = hist_df.dropna(
            subset=["dp", "di"]
        )

        # ----------------------------------------
        # EMPTY CHECK
        # ----------------------------------------

        if hist_df.empty:

            st.warning(
                "Fairness history contains no valid numeric values."
            )

        else:

            # ----------------------------------------
            # CREATE INDEX
            # ----------------------------------------

            hist_df = hist_df.reset_index(drop=True)

            hist_df["step"] = hist_df.index + 1

            # ----------------------------------------
            # PLOTLY FIGURE
            # ----------------------------------------
            
            fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    hist_df["step"],
    hist_df["dp"],
    marker="o",
    label="DP Difference"
)

ax.plot(
    hist_df["step"],
    hist_df["di"],
    marker="o",
    label="DI Ratio"
)

ax.set_title("Fairness Trends Over Time")

ax.set_xlabel("Run")

ax.set_ylabel("Metric Value")

ax.legend()

st.pyplot(fig)

plt.close(fig)
# ------------------------------------------------
# HUMAN REVIEW
# ------------------------------------------------

st.header("Human Review")
clean_sample = sample.to_frame(name="Value").reset_index()
clean_sample.columns = ["Feature", "Value"]

clean_sample["Value"] = pd.to_numeric(clean_sample["Value"], errors="coerce")

st.dataframe(clean_sample.fillna(0))

sample_df = sample.to_frame().T
sample_df = sample_df[X_test.columns]
sample_df = pd.DataFrame([sample])[X_test.columns]
sample_df = enforce_numeric(sample_df)
sample_df = sample_df.apply(pd.to_numeric, errors='coerce').fillna(0)
prediction = model.predict(sample_df)[0]

prob = model.predict_proba(sample_df)[0][1]

st.write(f"Prediction: {prediction} | Confidence: {round(prob,3)}")

decision = st.radio("Review", ["Accept", "Override"])

if decision == "Override":

    corrected = st.radio("Correct Label", ["Income > 50K","Income ≤ 50K"])

    log_human_feedback(sample_index, prediction, corrected)
    st.success("Saved")

feedback_path = os.path.join(BASE_DIR, "data", "human_feedback.csv")

if os.path.exists(feedback_path):
    st.subheader("Feedback Log")
    st.dataframe(clean_dataframe(pd.read_csv(feedback_path).reset_index(drop=True)))

st.divider()

# ------------------------------------------------
# WHAT-IF SIMULATOR
# ------------------------------------------------

st.header("What-if Simulator")

age = st.slider("Age", 18, 90, 35)
hours = st.slider("Hours", 1, 80, 40)
education = st.slider("Education", 1, 16, 10)
capital_gain = st.slider("Capital Gain", 0, 100000, 0)
sex = st.selectbox("Gender", ["Female","Male"])

sim_index = st.slider("Base Case", 0, len(X_test)-1)
sim_sample = X_test.iloc[sim_index].copy()

sim_sample["age"] = age
sim_sample["hours_per_week"] = hours
sim_sample["education_num"] = education
sim_sample["capital_gain"] = capital_gain
sim_sample["sex_Male"] = 1 if sex == "Male" else 0

sim_df = pd.DataFrame([sim_sample])[X_test.columns]
sim_df = enforce_numeric(sim_df)

sim_df = sim_df.apply(pd.to_numeric, errors='coerce').fillna(0)

prob = model.predict_proba(sim_df)[0][1]

threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5)

pred = 1 if prob > threshold else 0

st.write(f"Probability of >50K: {round(prob,2)}")
confidence = prob if pred == 1 else (1 - prob)
st.write("Prediction:", pred, "| Confidence:", round(confidence,2))

counter = sim_sample.copy()
counter["sex_Male"] = 1 - counter["sex_Male"]

counter_df = pd.DataFrame([counter])[X_test.columns]
counter_df = enforce_numeric(counter_df)
counter_df = counter_df[X_test.columns]

counter_df = counter_df.apply(pd.to_numeric, errors='coerce').fillna(0)

counter_prob = model.predict_proba(counter_df)[0][1]
counter_pred = 1 if counter_prob > threshold else 0
if counter_pred != pred:
    st.error("Bias risk")
else:
    st.success("Fair")