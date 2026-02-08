"""
Week 4 — Model Deployment (Model-as-a-Service)

Project : FactoryGuard AI

Goal    : Serve failure probability + SHAP explanation via REST API with Frontend UI

Author  : Rohn Dhwan (Adapted for local deployment)
"""

# =====================================
# 0. IMPORTS & SETUP
# =====================================

import json
import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify, render_template

# =====================================
# 1. LOAD MODEL & METADATA
# =====================================

MODEL_PATH = "models/factoryguard_xgb.pkl"
FEATURE_PATH = "models/feature_columns.pkl"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

explainer = shap.TreeExplainer(model)

# =====================================
# 2. FLASK APP
# =====================================

app = Flask(__name__)

# =====================================
# 3. HEALTH CHECK
# =====================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "UP"}), 200

# =====================================
# 4. HOME PAGE (FRONTEND)
# =====================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# =====================================
# 5. PREDICTION ENDPOINT
# =====================================

@app.route("/predict", methods=["POST"])
def predict():
    input_json = request.get_json()

    # --- Validation ---
    missing = [f for f in feature_columns if f not in input_json]
    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing
        }), 400

    # --- Create DataFrame ---
    input_df = pd.DataFrame([input_json])[feature_columns]

    # --- Prediction ---
    failure_prob = float(model.predict_proba(input_df)[0, 1])

    # --- SHAP Explanation ---
    shap_values = explainer(input_df)
    shap_contrib = [float(x) for x in shap_values.values[0]]

    shap_dict = dict(
        sorted(
            zip(feature_columns, shap_contrib),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
    )

    # --- Response ---
    return jsonify({
        "failure_probability": round(failure_prob, 4),
        "top_risk_factors": shap_dict
    })

# =====================================
# 6. ENTRY POINT
# =====================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
