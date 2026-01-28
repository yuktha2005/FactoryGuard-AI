# src/data/data_ingestion.py

import pandas as pd
from ..config import RAW_DATA_PATH, TIMESTAMP_COL

def load_data(path=RAW_DATA_PATH):
    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL)
    return df
