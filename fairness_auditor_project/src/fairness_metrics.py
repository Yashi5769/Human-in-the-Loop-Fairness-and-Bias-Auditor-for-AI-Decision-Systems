import pandas as pd


def demographic_parity_difference(predictions, sensitive_feature):

    df = pd.DataFrame({
        "prediction": predictions,
        "group": sensitive_feature
    })

    rates = df.groupby("group")["prediction"].mean()

    return rates.max() - rates.min()


def disparate_impact_ratio(predictions, sensitive_feature):

    df = pd.DataFrame({
        "prediction": predictions,
        "group": sensitive_feature
    })

    rates = df.groupby("group")["prediction"].mean()

    return rates.min() / rates.max()


def group_outcome_rates(predictions, sensitive_feature):

    df = pd.DataFrame({
        "prediction": predictions,
        "group": sensitive_feature
    })

    rates = df.groupby("group")["prediction"].mean()

    return rates