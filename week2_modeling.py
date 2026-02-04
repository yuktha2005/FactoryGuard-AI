import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, classification_report

from xgboost import XGBClassifier


# =====================================
# 1. LOAD PROCESSED DATASET
# =====================================

processed_path = "Data/Processed/sensor_data_engineered.csv"
df = pd.read_csv(processed_path)

print("Dataset shape:", df.shape)


# =====================================
# 2. DEFINE FEATURES & TARGET
# =====================================

target_column = "label"

X = df.drop(columns=[target_column])
y = df[target_column]

# DROP NON-NUMERIC COLUMNS (timestamps, strings)
non_numeric_cols = X.select_dtypes(include=["object"]).columns
print("Dropping non-numeric columns:", list(non_numeric_cols))

X = X.drop(columns=non_numeric_cols)

# =====================================
# 3. TRAIN–TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================
# 4. RANDOM FOREST (BASELINE MODEL)
# =====================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\n===== RANDOM FOREST RESULTS =====")
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)
print(classification_report(y_test, rf_pred))


# =====================================
# 5. HANDLE CLASS IMBALANCE FOR XGBOOST
# =====================================

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])


# =====================================
# 6. XGBOOST (HIGH-DIMENSIONAL MODEL)
# =====================================

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)


# =====================================
# 7. THRESHOLD-BASED PREDICTION
# =====================================

y_prob = xgb_model.predict_proba(X_test)[:, 1]

threshold = 0.3   # lower threshold → higher recall
xgb_pred = (y_prob >= threshold).astype(int)


# =====================================
# 8. XGBOOST EVALUATION
# =====================================

xgb_recall = recall_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)

print("\n===== XGBOOST RESULTS =====")
print("Recall:", xgb_recall)
print("F1 Score:", xgb_f1)
print(classification_report(y_test, xgb_pred))


# =====================================
# 9. FINAL COMPARISON
# =====================================

print("\n===== FINAL COMPARISON =====")
print(f"Random Forest -> Recall: {rf_recall:.4f}, F1: {rf_f1:.4f}")
print(f"XGBoost       -> Recall: {xgb_recall:.4f}, F1: {xgb_f1:.4f}")
