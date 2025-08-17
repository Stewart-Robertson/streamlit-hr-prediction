import pandas as pd
from pathlib import Path

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw HR attrition dataset."""
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning transformations to HR attrition dataset."""
    
    # 1. Drop unique identifier (employee_id)
    if "employee_id" in df.columns:
        df = df.drop(columns=["employee_id"])

    # 2. Handle duplicates
    df = df.drop_duplicates()

    # 3. Impute missing values
    # - engagement_score, manager_quality (continuous)
    if "engagement_score" in df.columns:
        df["engagement_score"] = df["engagement_score"].fillna(df["engagement_score"].median())
    if "manager_quality" in df.columns:
        df["manager_quality"] = df["manager_quality"].fillna(df["manager_quality"].median())

    # 4. Encode categorical correlations (role vs department)
    if "role" in df.columns and "department" in df.columns:
        df = df.drop(columns=["role"])  # keep department, drop role

    # 5. Drop highly correlated redundant fields
    drop_cols = []
    if "months_since_hire" in df.columns and "tenure_years" in df.columns:
        drop_cols.append("months_since_hire")
    if "salary_band" in df.columns and "base_salary" in df.columns:
        drop_cols.append("salary_band")
    if "exit_interview_scheduled" in df.columns and "attrited" in df.columns:
        drop_cols.append("exit_interview_scheduled")
    if "offboarding_ticket_created" in df.columns and "attrited" in df.columns:
        drop_cols.append("offboarding_ticket_created")

    df = df.drop(columns=drop_cols, errors="ignore")

    return df

def save_clean_data(df: pd.DataFrame, outpath: str):
    """Save cleaned dataset."""
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)

def main():
    raw_fp = "data/raw/hr_attrition.csv"
    clean_fp = "data/processed/hr_attrition_clean.csv"

    df_raw = load_raw_data(raw_fp)
    df_clean = clean_data(df_raw)
    save_clean_data(df_clean, clean_fp)

if __name__ == "__main__":
    main()