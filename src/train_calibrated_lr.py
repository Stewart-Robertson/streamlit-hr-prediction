# src/train_calibrated_lr.py
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------
# CONFIG
# -----------------------
DATA_PATH = Path("../data/processed/hr_attrition_clean.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "attrited"
DATE_COL = "snapshot_date"          # raw date (if present)
YEAR_COL = "snapshot_year"          # derived year column used for splitting
TRAIN_MAX_YEAR = 2022               # train ≤ this year; calibrate/test on next year
SKLEARN_VERSION_TAG = "skl171"      # for filename clarity

DOMAIN_LEAKAGE_DROPS = [
    "employee_id",
    "exit_interview_scheduled",
    "offboarding_ticket_created",
    "months_since_hire",
    "salary_band",
    DATE_COL,
    YEAR_COL,
]

VIF_INFORMED_DROPS = [
    "compa_ratio",
    "avg_raise_3y",
    "benefit_score",
    "manager_quality",          
    "age",
    "internal_moves_last_2y",
]

# -----------------------
# UTILITIES
# -----------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Ensure YEAR_COL exists without mutating DATE_COL semantics
    if YEAR_COL not in df.columns:
        if DATE_COL in df.columns:
            # Parse date column if needed and derive year
            if not np.issubdtype(df[DATE_COL].dtype, np.datetime64):
                df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
            df[YEAR_COL] = df[DATE_COL].dt.year
        else:
            # Try alternative date-like columns
            alt_dates = [c for c in ["snapshot_dt", "asof_date", "date"] if c in df.columns]
            if alt_dates:
                dcol = alt_dates[0]
                if not np.issubdtype(df[dcol].dtype, np.datetime64):
                    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
                df[YEAR_COL] = df[dcol].dt.year
            elif "year" in df.columns:
                df[YEAR_COL] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            else:
                raise ValueError(f"Could not find {YEAR_COL} or any snapshot date/year column.")
    return df

def add_mgmt_workload_features(df: pd.DataFrame, mq_col: str = "manager_quality", wl_col: str = "workload_score") -> pd.DataFrame:
    """
    Create interpretable binary flag and continuous interaction without leakage.
    - Missing manager quality is imputed to 1 (worst possible).
    - Binary flag (mgmt_workload_risk) is for stakeholder insights (will be dropped from the model).
    - Continuous interaction (mgmt_workload_score) is used by the model; scaling handled in pipeline.
    If required columns are missing, this is a no-op.
    """
    if mq_col not in df.columns or wl_col not in df.columns:
        return df

    mq = df[mq_col].fillna(1)
    wl = df[wl_col]

    # Binary flag for insights (kept in dataframe, dropped from modeling features)
    df["mgmt_workload_risk"] = ((df[mq_col].isna()) | ((mq < 5) & (wl > 7))).astype(int)

    # Continuous interaction for modeling (no external scaling here)
    df["mgmt_workload_score"] = (10 - mq) * wl
    return df

def make_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the modeling frame:
    1) Start from the raw df.
    2) Add engineered features that depend on manager_quality/workload_score.
    3) Drop leakage/identifier/year and VIF-informed columns (incl. manager_quality itself).
    4) Remove the binary insight flag from modeling features.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL} not found.")

    # 1–2) Add engineered features BEFORE dropping source columns
    dfm = df.copy()
    dfm = add_mgmt_workload_features(dfm)

    # 3) Drop leakage/identifier/year + VIF-informed
    drop_cols = [c for c in (DOMAIN_LEAKAGE_DROPS + VIF_INFORMED_DROPS) if c in dfm.columns]
    dfm = dfm.drop(columns=drop_cols, errors="ignore")

    # 4) Do not feed the binary risk flag into the model
    if "mgmt_workload_risk" in dfm.columns:
        dfm = dfm.drop(columns=["mgmt_workload_risk"])

    # Split X/y
    X = dfm.drop(columns=[TARGET_COL], errors="ignore")
    y = df[TARGET_COL].astype(int)

    # Safety check: ensure engineered continuous feature exists if source cols existed
    if {"manager_quality", "workload_score"}.issubset(set(df.columns)):
        assert "mgmt_workload_score" in X.columns, (
            "Expected engineered feature 'mgmt_workload_score' to exist. "
            "Ensure feature engineering runs before drops."
        )
    return X, y

def split_time(X: pd.DataFrame, y: pd.Series, years: pd.Series, train_max_year: int):
    train_mask = years <= train_max_year
    cal_mask = years == (train_max_year + 1)
    if cal_mask.sum() == 0:
        raise ValueError(f"No rows for calibration/test year {train_max_year + 1}.")
    return (
        X.loc[train_mask], y.loc[train_mask],
        X.loc[cal_mask],   y.loc[cal_mask],
    )

def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    return pre

def fit_base_lr(pre: ColumnTransformer, X_tr: pd.DataFrame, y_tr: pd.Series) -> Pipeline:
    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_tr, y_tr)
    return pipe

def calibrate(pipe_fitted: Pipeline, X_cal: pd.DataFrame, y_cal: pd.Series):
    # Try both methods; pick lower Brier score
    def _fit(method: str):
        cal = CalibratedClassifierCV(pipe_fitted, method=method, cv="prefit")
        cal.fit(X_cal, y_cal)
        proba = cal.predict_proba(X_cal)[:, 1]
        metrics = {
            "method": method,
            "brier": brier_score_loss(y_cal, proba),
            "roc_auc": roc_auc_score(y_cal, proba),
            "pr_auc": average_precision_score(y_cal, proba),
        }
        return cal, metrics

    cal_sig, m_sig = _fit("sigmoid")
    cal_iso, m_iso = _fit("isotonic")

    chosen = (cal_iso, m_iso) if m_iso["brier"] < m_sig["brier"] else (cal_sig, m_sig)
    return chosen

def save_artifacts(model, metrics: dict, tag: str):
    model_path = MODEL_DIR / f"attrition_lr_calibrated_train_to_{TRAIN_MAX_YEAR}_{tag}.pkl"
    joblib.dump(model, model_path)
    (MODEL_DIR / f"attrition_lr_calibrated_metrics_train_to_{TRAIN_MAX_YEAR}_{tag}.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(f"Saved model → {model_path}")
    print(f"Saved metrics → {model_path.with_suffix('.json')}")

# -----------------------
# MAIN
# -----------------------
def main():
    df = load_data(DATA_PATH)
    X_all, y_all = make_feature_frame(df)
    X_tr, y_tr, X_cal, y_cal = split_time(X_all, y_all, df[YEAR_COL].astype(int), TRAIN_MAX_YEAR)

    pre = build_preprocess(X_tr)
    base = fit_base_lr(pre, X_tr, y_tr)

    calibrated_model, cal_metrics = calibrate(base, X_cal, y_cal)
    print("Calibration metrics:", cal_metrics)

    save_artifacts(calibrated_model, cal_metrics, SKLEARN_VERSION_TAG)

if __name__ == "__main__":
    main()