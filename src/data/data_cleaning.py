# src/data/data_cleaning.py

from ..config import SENSOR_COLS, TIMESTAMP_COL

def clean_data(df):
    df = df.set_index(TIMESTAMP_COL)

    # Time-aware interpolation (NO future leakage)
    df[SENSOR_COLS] = df[SENSOR_COLS].interpolate(
        method="time",
        limit_direction="forward"
    )

    return df.reset_index()
