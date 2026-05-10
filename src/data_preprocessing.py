import pandas as pd
import streamlit as st

COLUMN_NAMES = [
    "age","workclass","fnlwgt","education","education_num",
    "marital_status","occupation","relationship","race","sex",
    "capital_gain","capital_loss","hours_per_week","native_country","income"
]

@st.cache_data

def load_data(path):

    df = pd.read_csv(
        path,
        names=COLUMN_NAMES,
        na_values=" ?",
        skipinitialspace=True
    )

    return df


def dataset_overview(df):

    print("\nDataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nIncome Distribution:")
    print(df["income"].value_counts())


def preprocess_data(df):

    df = df.dropna()

    df["income"] = df["income"].apply(
        lambda x: 1 if ">50K" in x else 0
    )

    sensitive_feature = df["sex"]

    categorical_cols = df.select_dtypes(include="object").columns

    df_encoded = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    return df_encoded, sensitive_feature