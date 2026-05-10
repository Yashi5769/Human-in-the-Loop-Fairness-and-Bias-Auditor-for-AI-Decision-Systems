import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def identify_worst_group(df, sensitive_cols):

    df = df.copy()

    # Ensure prediction column exists
    if "prediction" not in df.columns:
        raise ValueError("DataFrame must contain 'prediction' column")

    # Remove missing rows safely
    df = df.dropna(subset=sensitive_cols + ["prediction"])

    # Create grouped fairness rates
    group_rates = (
        df.groupby(sensitive_cols)["prediction"]
        .mean()
        .sort_values()
    )

    # Safety check
    if len(group_rates) == 0:
        return "Unknown", pd.Series(dtype=float)

    # Worst group = lowest positive outcome rate
    worst_group = group_rates.idxmin()

    return worst_group, group_rates


def apply_reweighting(df, worst_group, sensitive_cols):

    df = df.copy()

    # recreate group column
    df["group"] = df[sensitive_cols].astype(str).agg(" | ".join, axis=1)

    # assign higher weight to worst group
    df["weights"] = np.where(df["group"] == worst_group, 4.0, 1.0)

    return df


def retrain_with_weights(df, target_col="income"):

    df = df.copy()

    # -----------------------------------
    # DROP NON-FEATURE COLUMNS
    # -----------------------------------
    drop_cols = [
        target_col,
        "weights",
        "group",
        "gender",
        "race"
    ]

    X = df.drop(
        columns=[col for col in drop_cols if col in df.columns]
    )

    y = df[target_col]

    # default weights if missing
    if "weights" not in df.columns:
        df["weights"] = 1.0

    w = df["weights"]

    # -----------------------------------
    # FORCE NUMERIC
    # -----------------------------------
    X = X.apply(pd.to_numeric, errors="coerce")

    # -----------------------------------
    # HANDLE MISSING VALUES
    # -----------------------------------
    imputer = SimpleImputer(strategy="median")

    X = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )

    # -----------------------------------
    # TRAIN TEST SPLIT
    # -----------------------------------
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y,
        w,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------------
    # TRAIN MODEL
    # -----------------------------------
    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=w_train
    )

    preds = model.predict(X_test)

    return model, X_test, y_test, preds