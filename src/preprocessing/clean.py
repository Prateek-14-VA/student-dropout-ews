"""
Data Cleaning Script — UCI #697
Fixes column names, verifies dtypes, and saves a clean version.
"""
import os
import pandas as pd
import numpy as np


def load_raw():
    """Load the raw dataset from data/raw."""
    path = 'data/raw/uci_697_dropout.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, sep=';')


def clean_column_names(df):
    """Strip whitespace and tabs from column names."""
    original = df.columns.tolist()
    df.columns = df.columns.str.strip().str.replace('\t', '', regex=False)
    fixed = [(o, n) for o, n in zip(original, df.columns) if o != n]
    if fixed:
        print("\n[FIX] Column names corrected:")
        for old, new in fixed:
            print(f"   {repr(old)} -> {repr(new)}")
    else:
        print("\n[OK] All column names clean")
    return df


def clean_target(df):
    """Strip whitespace from the Target column."""
    df['Target'] = df['Target'].astype(str).str.strip()
    print(f"\n[TARGET] Unique values: {df['Target'].unique().tolist()}")
    print(f"Distribution:\n{df['Target'].value_counts()}")
    return df


def verify_dtypes(df):
    """Report dtypes and flag any unexpected object columns."""
    print("\n[DTYPES] Summary:")
    print(df.dtypes.value_counts())

    obj_cols = df.select_dtypes(include='object').columns.tolist()
    print(f"\nObject columns: {obj_cols}")
    return df


def save_clean(df, output_path):
    """Save cleaned DataFrame."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[SAVED] {output_path}  (shape={df.shape})")


def main():
    print("=" * 60)
    print("CLEANING - UCI #697")
    print("=" * 60)

    df = load_raw()
    print(f"\nLoaded shape: {df.shape}")

    df = clean_column_names(df)
    df = clean_target(df)
    df = verify_dtypes(df)

    save_clean(df, 'data/processed/uci_697_clean.csv')

    print("\n" + "=" * 60)
    print("[DONE] CLEANING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()