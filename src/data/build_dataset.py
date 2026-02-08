# src/data/build_dataset.py

import os
from .data_ingestion import load_data
from .data_cleaning import clean_data
from .feature_engineering import (
    create_lag_features,
    create_rolling_features
)
from .leakage_checks import check_data_leakage
from ..config import PROCESSED_DATA_PATH

def run_pipeline():
    print("Loading raw sensor data...")
    df = load_data()

    print(" Cleaning missing values...")
    df = clean_data(df)

    print("Creating lag features...")
    df = create_lag_features(df)

    print("Creating rolling statistics...")
    df = create_rolling_features(df)

    print("Checking for data leakage...")
    check_data_leakage(df)

    print("Filling NaNs from lagging with 0...")
    df = df.fillna(0)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Week 1 dataset created → {PROCESSED_DATA_PATH}")
    print(f"Final shape: {df.shape}")

if __name__ == "__main__":
    run_pipeline()
