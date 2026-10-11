"""
Train-test split + class imbalance handling for UCI #697.
Produces stratified 80/20 splits and SMOTE-augmented training sets.
Categorical columns are label-encoded before SMOTE.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = 'Target_encoded'


def load_features(name):
    path = f'data/processed/{name}_features.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}. Run features.py first.")
    return pd.read_csv(path)


def encode_categoricals(df, name):
    """Label-encode every object column. Save encoders for later use."""
    print(f"\n[ENCODE] {name} — label encoding categoricals")
    encoders = {}
    obj_cols = df.select_dtypes(include='object').columns.tolist()

    for col in obj_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"   {col}: {len(le.classes_)} classes")

    os.makedirs('models/encoders', exist_ok=True)
    joblib.dump(encoders, f'models/encoders/{name}_encoders.pkl')
    print(f"   [SAVED] models/encoders/{name}_encoders.pkl")
    return df


def split(df, name):
    """Stratified split on Target_encoded."""
    print(f"\n[SPLIT] {name}")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"   Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"   Train class dist: {y_train.value_counts().sort_index().to_dict()}")
    print(f"   Test  class dist: {y_test.value_counts().sort_index().to_dict()}")

    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, name):
    """SMOTE on training data only."""
    print(f"\n[SMOTE] {name}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"   Before: {y_train.value_counts().sort_index().to_dict()}")
    print(f"   After : {y_res.value_counts().sort_index().to_dict()}")
    print(f"   New shape: {X_res.shape}")
    return X_res, y_res


def save_split(X_train, X_test, y_train, y_test, X_smote, y_smote, name):
    os.makedirs('data/processed', exist_ok=True)

    train = X_train.copy(); train[TARGET_COL] = y_train.values
    test = X_test.copy();   test[TARGET_COL]  = y_test.values
    smote_df = X_smote.copy(); smote_df[TARGET_COL] = y_smote.values

    train.to_csv(f'data/processed/{name}_train.csv', index=False)
    test.to_csv(f'data/processed/{name}_test.csv', index=False)
    smote_df.to_csv(f'data/processed/{name}_train_smote.csv', index=False)

    print(f"   [SAVED] {name}_train.csv        shape={train.shape}")
    print(f"   [SAVED] {name}_test.csv         shape={test.shape}")
    print(f"   [SAVED] {name}_train_smote.csv  shape={smote_df.shape}")


def process(name):
    df = load_features(name)
    df = encode_categoricals(df, name)
    X_train, X_test, y_train, y_test = split(df, name)
    X_smote, y_smote = apply_smote(X_train, y_train, name)
    save_split(X_train, X_test, y_train, y_test, X_smote, y_smote, name)


def main():
    print("=" * 60)
    print("TRAIN-TEST SPLIT + SMOTE (with label encoding)")
    print("=" * 60)

    process('admission')
    process('semester')

    print("\n" + "=" * 60)
    print("[DONE] SPLIT + SMOTE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()