import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text


def _load_csv(csv_path: str) -> pd.DataFrame:
    """Read the uploaded CSV into a DataFrame.
    The CSV is expected to contain a column named ``record_id`` that uniquely
    identifies each row. If it does not exist we generate one from the index.
    """
    df = pd.read_csv(csv_path)
    if 'record_id' not in df.columns:
        df.insert(0, 'record_id', df.index.astype(str))
    return df


def _engine_from_env():
    """Create a SQLAlchemy engine from environment variables.
    The same variables are used in ``lead_scoring_engine.settings``.
    """
    import os
    engine = create_engine(
        f"{os.getenv('DB_ENGINE', 'postgresql+psycopg2')}://"
        f"{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', '')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'lead_scoring')}"
    )
    return engine


def _sql_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Run window‑function based feature extraction against the DB.
    For demonstration we perform a simple aggregation on the ``customers``
    table that mirrors the uploaded CSV. In a real project the queries would be
    more sophisticated (rolling averages, counts, etc.).
    """
    engine = _engine_from_env()
    # Example: compute total purchases per customer in the past 30 days
    query = text(
        """
        SELECT
            c.record_id,
            SUM(o.amount) AS total_spent_last_30d,
            COUNT(o.id) AS purchase_count_last_30d
        FROM customers AS c
        LEFT JOIN orders AS o
          ON o.customer_id = c.id
         AND o.order_date >= CURRENT_DATE - INTERVAL '30 days'
        WHERE c.record_id = ANY(:ids)
        GROUP BY c.record_id
        """
    )
    ids = df['record_id'].tolist()
    result = pd.read_sql(query, engine, params={"ids": ids})
    return df.merge(result, on='record_id', how='left')


def preprocess(csv_path: str) -> pd.DataFrame:
    """Full preprocessing pipeline.
    1️⃣ Load raw CSV.
    2️⃣ Basic numeric cleaning (fill NaNs, type conversion).
    3️⃣ Optional SQL‑based feature engineering.
    4️⃣ Return a DataFrame ready for model input (numeric columns only).
    """
    df = _load_csv(csv_path)
    # Basic cleaning – ensure numeric columns are floats
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].astype(float).fillna(0)
    # Add example engineered features using pandas (if DB not required)
    if 'login_timestamp' in df.columns:
        df['days_since_login'] = (pd.Timestamp('now') - pd.to_datetime(df['login_timestamp'])).dt.days
    if 'order_amount' in df.columns:
        df['avg_order_value'] = df['order_amount']
    # Merge SQL‑derived features (if the table exists)
    try:
        df = _sql_feature_engineering(df)
    except Exception:
        # Silently ignore if DB or tables are missing – fallback to pandas only
        pass
    # Drop identifiers that are not model features
    feature_df = df.drop(columns=['record_id'], errors='ignore')
    return feature_df
