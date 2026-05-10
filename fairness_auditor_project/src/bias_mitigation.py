import pandas as pd


def reweigh_dataset(df, sensitive_attribute, target):
    """
    Apply reweighing technique to balance protected groups.

    Parameters
    ----------
    df : pandas DataFrame
    sensitive_attribute : protected attribute (e.g., 'sex')
    target : prediction label (e.g., 'income')

    Returns
    -------
    DataFrame with sample weights
    """

    df = df.copy()

    group_counts = df.groupby([sensitive_attribute, target]).size()

    total_samples = len(df)

    weights = {}

    for (group, label), count in group_counts.items():

        weights[(group, label)] = total_samples / (len(group_counts) * count)

    df["sample_weight"] = df.apply(
        lambda row: weights[(row[sensitive_attribute], row[target])],
        axis=1
    )

    return df