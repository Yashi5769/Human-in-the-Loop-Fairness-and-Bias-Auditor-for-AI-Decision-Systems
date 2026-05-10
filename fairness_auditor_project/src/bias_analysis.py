import pandas as pd


def shap_bias_analysis(shap_values, X_sample, sensitive_feature):

    shap_df = pd.DataFrame(
        shap_values.values,
        columns=X_sample.columns
    )

    shap_df["group"] = sensitive_feature.values

    group_means = shap_df.groupby("group").mean()

    bias_table = pd.DataFrame({
        "Male_Contribution": group_means.iloc[1],
        "Female_Contribution": group_means.iloc[0]
    })

    bias_table["Difference"] = abs(
        bias_table["Male_Contribution"] -
        bias_table["Female_Contribution"]
    )

    bias_table = bias_table.sort_values(
        by="Difference",
        ascending=False
    )

    return bias_table