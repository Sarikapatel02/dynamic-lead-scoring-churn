import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_models(df: pd.DataFrame, target: str = "churn"):
    """Train RandomForest and GradientBoosting classifiers.
    The input ``df`` must contain the target column (default ``churn``) and only
    numeric feature columns.
    Returns a dict with the two fitted models and their validation AUC.
    """
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_pred)

    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict_proba(X_val)[:, 1]
    gb_auc = roc_auc_score(y_val, gb_pred)

    # Persist models
    joblib.dump(rf, MODEL_DIR / "random_forest.joblib")
    joblib.dump(gb, MODEL_DIR / "gradient_boosting.joblib")

    return {
        "random_forest": rf,
        "gradient_boosting": gb,
        "rf_auc": rf_auc,
        "gb_auc": gb_auc,
    }

if __name__ == "__main__":
    # For quick CLI usage: python train.py path/to/training.csv
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py <training_csv_path>")
        sys.exit(1)
    data_path = sys.argv[1]
    df = pd.read_csv(data_path)
    # Expect a binary column named 'churn' (1=churn, 0=stay)
    results = train_models(df, target="churn")
    print(f"RandomForest AUC: {results['rf_auc']:.4f}")
    print(f"GradientBoosting AUC: {results['gb_auc']:.4f}")
    print(f"Models saved to {MODEL_DIR}")
