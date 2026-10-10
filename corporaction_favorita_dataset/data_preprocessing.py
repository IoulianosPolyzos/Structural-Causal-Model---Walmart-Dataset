import pandas as pd

def check_negatives(df):
    df = df.copy()

    numeric_df = df.select_dtypes(include=['number'])

    negative_percent = (numeric_df < 0).sum() / len(numeric_df) * 100

    return negative_percent


def remove_negatives(df, columns=None):
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=['number']).columns

    df = df[(df[columns] >= 0).all(axis=1)]

    return df


def check_outliers(df, columns=None):
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=['number']).columns

    outlier_percent = {}

    for col in columns:
        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        percent = (outliers / len(df)) * 100

        outlier_percent[col] = percent

    return pd.Series(outlier_percent)

def check_zeros(df):
    df = df.copy()

    numeric_df = df.select_dtypes(include=['number'])

    zero_percent = (numeric_df == 0).sum() / len(numeric_df) * 100

    return zero_percent