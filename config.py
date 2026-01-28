# src/config.py

RAW_DATA_PATH = "data/raw/predictive_maintenance_dataset.csv"
PROCESSED_DATA_PATH = "data/processed/modeling_dataset.csv"

TIMESTAMP_COL = "timestamp"
TARGET_COL = "failure"

SENSOR_COLS = ["vibration", "temperature", "pressure"]

LAG_STEPS = [1, 2]
ROLLING_WINDOWS = [1, 4, 8]  # hours

#python -m src.data.build_dataset