# src/data/feature_engineering.py

from ..config import SENSOR_COLS, LAG_STEPS, ROLLING_WINDOWS

def create_lag_features(df):
    for col in SENSOR_COLS:
        for lag in LAG_STEPS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def create_rolling_features(df):
    for col in SENSOR_COLS:
        for window in ROLLING_WINDOWS:
            df[f"{col}_roll_mean_{window}h"] = (
                df[col].rolling(window=window, min_periods=1).mean()
            )
            df[f"{col}_roll_std_{window}h"] = (
                df[col].rolling(window=window, min_periods=1).std()
            )
    return df
