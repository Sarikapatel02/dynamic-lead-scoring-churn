import os
import joblib
import pandas as pd
import numpy as np
import shap
from .data_preprocess import preprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Load models (assumes they have been trained and saved)
RF_MODEL_PATH = MODEL_DIR / "random_forest.joblib"
GB_MODEL_PATH = MODEL_DIR / "gradient_boosting.joblib"

rf_model = joblib.load(RF_MODEL_PATH) if RF_MODEL_PATH.exists() else None
gb_model = joblib.load(GB_MODEL_PATH) if GB_MODEL_PATH.exists() else None

def _shap_explain(model, X):
    """Compute SHAP values for a tree‑based model.
    Returns a DataFrame where each column is a feature and each row is a record.
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    # For binary classification, shap_vals is a list with two arrays – take the one for class 1
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_vals = shap_vals[1]
    return pd.DataFrame(shap_vals, columns=X.columns)

def run_prediction_pipeline(csv_path: str):
    """End‑to‑end pipeline used by the Django view.
    1️⃣ Preprocess the input CSV (including optional SQL feature engineering).
    2️⃣ Generate predictions from both models.
    3️⃣ Compute SHAP contributions (using the Gradient Boosting model – usually more stable).
    4️⃣ Assemble a result DataFrame with record_id, probabilities, risk level and a JSON mapping of SHAP values.
    """
    # Load raw CSV to keep the record identifier
    raw_df = pd.read_csv(csv_path)
    if 'record_id' not in raw_df.columns:
        raw_df.insert(0, 'record_id', raw_df.index.astype(str))

    # Preprocess features (numeric only)
    X = preprocess(csv_path)

    # Ensure models are available
    if rf_model is None or gb_model is None:
        raise RuntimeError("Models not found. Train them first using ml_pipeline/train.py")

    # Predict churn probability (using Random Forest) and lead score (using Gradient Boosting)
    churn_prob = rf_model.predict_proba(X)[:, 1]
    lead_score = gb_model.predict_proba(X)[:, 1]

    # Determine risk level based on churn probability thresholds
    def risk_level(p):
        if p >= 0.8:
            return "High"
        elif p >= 0.5:
            return "Medium"
        else:
            return "Low"

    risks = [risk_level(p) for p in churn_prob]

    # SHAP explanations – we use Gradient Boosting model for consistency with lead score
    shap_df = _shap_explain(gb_model, X)
    # Convert each row's SHAP dict to JSON‑serialisable dict
    shap_json = shap_df.apply(lambda row: row.to_dict(), axis=1)

    results = pd.DataFrame({
        "record_id": raw_df["record_id"],
        "churn_prob": churn_prob,
        "lead_score": lead_score,
        "risk_level": risks,
        "shap_values": shap_json,
    })
    return results
