"""
Feature Engineering for UCI #697.
Builds two feature sets:
  1. Admission-stage: data available at admission time only
  2. Semester-stage: admission + first & second semester performance
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# Columns available at admission
ADMISSION_COLS = [
    'Marital status',
    'Application mode',
    'Application order',
    'Course',
    'Daytime/evening attendance',
    'Previous qualification',
    'Previous qualification (grade)',
    'Nacionality',
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    'Admission grade',
    'Displaced',
    'Educational special needs',
    'Debtor',
    'Tuition fees up to date',
    'Gender',
    'Scholarship holder',
    'Age at enrollment',
    'International',
    'Unemployment rate',
    'Inflation rate',
    'GDP',
]

# Columns only available after first & second semester
SEMESTER_COLS = [
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
]

TARGET_COL = 'Target_encoded'


def load_encoded():
    path = 'data/processed/uci_697_encoded.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run encode.py first. Missing: {path}")
    return pd.read_csv(path)


def check_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    print(f"   ✓ All {len(required)} columns present")


def build_admission(df):
    print("\n[BUILD] Admission-stage feature set")
    check_columns(df, ADMISSION_COLS + [TARGET_COL])
    cols = ADMISSION_COLS + [TARGET_COL]
    return df[cols].copy()


def build_semester(df):
    print("\n[BUILD] Semester-stage feature set")
    all_cols = ADMISSION_COLS + SEMESTER_COLS
    check_columns(df, all_cols + [TARGET_COL])
    cols = all_cols + [TARGET_COL]
    return df[cols].copy()


def save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"   [SAVED] {path}  (shape={df.shape})")


def main():
    print("=" * 60)
    print("FEATURE ENGINEERING - UCI #697")
    print("=" * 60)

    df = load_encoded()
    print(f"\nLoaded shape: {df.shape}")

    admission = build_admission(df)
    save(admission, 'data/processed/admission_features.csv')

    semester = build_semester(df)
    save(semester, 'data/processed/semester_features.csv')

    print("\n" + "=" * 60)
    print("[DONE] FEATURE ENGINEERING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()