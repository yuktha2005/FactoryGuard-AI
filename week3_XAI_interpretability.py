import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# =====================================
# 0. SETUP
# =====================================

os.makedirs("results/shap", exist_ok=True)
np.random.seed(42)

# =====================================
# 1. LOAD DATA
# =====================================

df = pd.read_csv("Data/Processed/sensor_data_engineered.csv")

TARGET = "label"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X = X.select_dtypes(exclude=["object"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# =====================================
# 2. TRAIN XGBOOST MODEL
# =====================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# =====================================
# 3. SHAP EXPLAINER (MODEL-AGNOSTIC)
# =====================================

background = X_train.sample(100, random_state=42)

X_test_shap = X_test.sample(100, random_state=42)
y_test_shap = y_test.loc[X_test_shap.index]

explainer = shap.Explainer(
    model.predict_proba,
    background
)

shap_values = explainer(X_test_shap)

# =====================================
# 4. SHAP SUMMARY PLOT (GLOBAL)
# =====================================

plt.figure()
shap.summary_plot(
    shap_values[:, :, 1],
    X_test_shap,
    show=False
)
plt.title("SHAP Summary Plot – Failure Risk")
plt.savefig("results/shap/shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# =====================================
# 5. SHAP FORCE PLOT (LOCAL – CORRECT)
# =====================================

failure_rows = y_test_shap[y_test_shap == 1]

if len(failure_rows) == 0:
    raise ValueError("No failure samples in SHAP subset")

row_index = failure_rows.index[0]
row_number = X_test_shap.index.get_loc(row_index)

base_value = shap_values.base_values[row_number, 1]
shap_row = shap_values[row_number, :, 1].values

shap.force_plot(
    base_value,
    shap_row,
    X_test_shap.iloc[row_number],
    matplotlib=True,
    show=False
)

plt.title("SHAP Force Plot – Individual Failure Prediction")
plt.savefig("results/shap/shap_force.png", dpi=300, bbox_inches="tight")
plt.close()

# =====================================
# 6. FEATURE IMPORTANCE TABLE
# =====================================

mean_abs_shap = np.abs(shap_values[:, :, 1].values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X_test_shap.columns,
    "Mean |SHAP Value|": mean_abs_shap
}).sort_values(by="Mean |SHAP Value|", ascending=False)

importance_df.to_csv(
    "results/shap/shap_feature_importance.csv",
    index=False
)

print("\nTop 10 Important Features:")
print(importance_df.head(10))

print("\n[Week-3 XAI completed successfully]")
