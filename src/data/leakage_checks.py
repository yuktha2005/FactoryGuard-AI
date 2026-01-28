# src/data/leakage_checks.py

def check_data_leakage(df):
    forbidden = ["future", "lead", "target_shift"]
    for col in df.columns:
        if any(word in col.lower() for word in forbidden):
            raise ValueError(f" Data leakage detected: {col}")
    print("No data leakage detected")
