import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_shap_values(model, X):
    import shap
    explainer = shap.Explainer(model, X)
    return explainer(X)

def compute_shap_explainer(model, X):
    explainer = shap.Explainer(model, X)
    return explainer


def get_shap_values(explainer, X):
    shap_values = explainer(X)
    return shap_values


def plot_local_explanation(shap_values, X, index):

    values = shap_values[index]

    fig, ax = plt.subplots(figsize=(8, 4))

    shap.plots.waterfall(values, show=False)

    return fig

def plot_shap_summary(shap_values, X):
    import shap
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    if shap_values is None or len(shap_values) == 0:
        return plt.figure()

    fig = plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    return fig

def plot_feature_contribution_bar(
    shap_values,
    X_sample,
    index,
    top_n=12
):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # -----------------------------------
    # GET SHAP VALUES
    # -----------------------------------

    values = shap_values.values[index]

    # multiclass safety
    if len(values.shape) > 1:
        values = values[:, 0]

    features = X_sample.iloc[index]

    # -----------------------------------
    # BUILD DATAFRAME
    # -----------------------------------

    contribution_df = pd.DataFrame({
        "Feature": X_sample.columns,
        "SHAP Value": values
    })

    # absolute importance
    contribution_df["ABS"] = (
        contribution_df["SHAP Value"]
        .abs()
    )

    # keep only top features
    contribution_df = (
        contribution_df
        .sort_values("ABS", ascending=False)
        .head(top_n)
    )

    # sort for horizontal bar chart
    contribution_df = contribution_df.sort_values(
        "SHAP Value"
    )

    # -----------------------------------
    # PLOT
    # -----------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [
        "green" if v > 0 else "red"
        for v in contribution_df["SHAP Value"]
    ]

    ax.barh(
        contribution_df["Feature"],
        contribution_df["SHAP Value"],
        color=colors
    )

    ax.set_title(
        f"Top {top_n} Feature Contributions"
    )

    ax.set_xlabel("SHAP Impact")

    plt.tight_layout()

    return fig