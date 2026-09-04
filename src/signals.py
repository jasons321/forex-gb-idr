import pandas as pd


def add_percentile_rank(df, period=180, col="Close"):
    df = df.copy()
    df["Percentile_Rank"] = df[col].rolling(window=period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )
    return df


def add_zscore(df, period=180, col="Close"):
    df = df.copy()
    roll_mean = df[col].rolling(period).mean()
    roll_std = df[col].rolling(period).std()
    df["Zscore"] = (df[col] - roll_mean) / roll_std
    return df


def classify_signal(p, favorable=0.25, unfavorable=0.75):
    if pd.isna(p):
        return None
    elif p <= favorable:
        return "Favorable"
    elif p >= unfavorable:
        return "Unfavorable"
    else:
        return "Neutral"