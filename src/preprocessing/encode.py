"""
Encode UCI #697 — decode categorical codes to labels + encode target.
"""
import os
import sys
import pandas as pd

# Add project root to path so we can import mappings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.preprocessing.mappings import COLUMN_DECODERS, TARGET_MAP


def load_clean():
    path = 'data/processed/uci_697_clean.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run clean.py first. Missing: {path}")
    return pd.read_csv(path)


def decode_categoricals(df):
    """Replace numeric codes with human-readable labels."""
    print("\n[DECODE] Applying mappings...")
    for col, mapping in COLUMN_DECODERS.items():
        if col in df.columns:
            before = df[col].nunique()
            df[col] = df[col].map(mapping).fillna(df[col].astype(str))
            print(f"   {col}: {before} codes -> labels")
    return df


def encode_target(df):
    """Map target labels to integers."""
    print(f"\n[TARGET] Encoding using: {TARGET_MAP}")
    df['Target_encoded'] = df['Target'].map(TARGET_MAP)
    unmapped = df[df['Target_encoded'].isna()]['Target'].unique().tolist()
    if unmapped:
        raise ValueError(f"Unmapped target values: {unmapped}")
    print(f"   Encoded distribution:\n{df['Target_encoded'].value_counts().sort_index()}")
    return df


def save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n[SAVED] {path}  (shape={df.shape})")


def main():
    print("=" * 60)
    print("ENCODING - UCI #697")
    print("=" * 60)

    df = load_clean()
    print(f"\nLoaded shape: {df.shape}")

    df = decode_categoricals(df)
    df = encode_target(df)

    save(df, 'data/processed/uci_697_encoded.csv')

    print("\n" + "=" * 60)
    print("[DONE] ENCODING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()