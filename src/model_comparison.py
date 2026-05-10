from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import pandas as pd

from src.fairness_metrics import (
    demographic_parity_difference,
    disparate_impact_ratio
)


def train_models_and_compare(df):

    X = df.drop("income", axis=1)
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            eval_metric='logloss',
            random_state=42
        )
    }

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        sex_feature = X_test["sex_Male"]

        dp = demographic_parity_difference(
            preds,
            sex_feature
        )

        di = disparate_impact_ratio(
            preds,
            sex_feature
        )

        results.append({
            "Model": str(name),
            "Accuracy": float(acc),
            "DP Difference": float(abs(dp)),
            "DI Ratio": float(di)
        })

    return results


def recommend_best_model(results, alpha=0.5, beta=0.5):

    df = pd.DataFrame(results)

    # -----------------------------
    # FORCE CLEAN NUMERIC TYPES
    # -----------------------------

    numeric_cols = [
        "Accuracy",
        "DP Difference",
        "DI Ratio"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.fillna(0)

    # -----------------------------
    # FAIRNESS SCORE
    # -----------------------------

    df["Fairness Score"] = (
        1 - df["DP Difference"]
    )

    # clip between 0 and 1
    df["Fairness Score"] = (
        df["Fairness Score"]
        .clip(0, 1)
    )

    # -----------------------------
    # FINAL SCORE
    # -----------------------------

    df["Final Score"] = (
        alpha * df["Accuracy"] +
        beta * df["Fairness Score"]
    )

    # -----------------------------
    # ROUNDING
    # -----------------------------

    df["Accuracy"] = df["Accuracy"].round(3)
    df["DP Difference"] = df["DP Difference"].round(3)
    df["DI Ratio"] = df["DI Ratio"].round(3)
    df["Fairness Score"] = df["Fairness Score"].round(3)
    df["Final Score"] = df["Final Score"].round(3)

    # -----------------------------
    # BEST MODEL
    # -----------------------------

    best_model = df.loc[
        df["Final Score"].idxmax()
    ]

    return df, best_model