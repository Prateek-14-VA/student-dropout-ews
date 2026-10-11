"""
Unit tests for preprocessing pipeline.
Run with: pytest tests/test_preprocessing.py -v
"""
import os
import sys
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.pipeline import transform_new_student


DATA = 'data/processed'
TARGET_COL = 'Target_encoded'


def test_clean_file_exists():
    assert os.path.exists(f'{DATA}/uci_697_clean.csv')


def test_encoded_file_exists():
    assert os.path.exists(f'{DATA}/uci_697_encoded.csv')


def test_feature_files_exist():
    assert os.path.exists(f'{DATA}/admission_features.csv')
    assert os.path.exists(f'{DATA}/semester_features.csv')


def test_clean_shape():
    df = pd.read_csv(f'{DATA}/uci_697_clean.csv')
    assert df.shape == (4424, 37), f"Expected (4424, 37), got {df.shape}"


def test_no_tabs_in_columns():
    df = pd.read_csv(f'{DATA}/uci_697_clean.csv')
    for col in df.columns:
        assert '\t' not in col, f"Tab found in column: {repr(col)}"


def test_encoded_target_values():
    df = pd.read_csv(f'{DATA}/uci_697_encoded.csv')
    assert TARGET_COL in df.columns
    assert set(df[TARGET_COL].unique()) == {0, 1, 2}


def test_admission_feature_count():
    df = pd.read_csv(f'{DATA}/admission_features.csv')
    assert df.shape == (4424, 25), f"Expected (4424, 25), got {df.shape}"


def test_semester_feature_count():
    df = pd.read_csv(f'{DATA}/semester_features.csv')
    assert df.shape == (4424, 37), f"Expected (4424, 37), got {df.shape}"


def test_train_test_no_overlap():
    for name in ('admission', 'semester'):
        train = pd.read_csv(f'{DATA}/{name}_train.csv')
        test = pd.read_csv(f'{DATA}/{name}_test.csv')
        assert list(train.columns) == list(test.columns)


def test_stratification_preserved():
    for name in ('admission', 'semester'):
        train = pd.read_csv(f'{DATA}/{name}_train.csv')
        test = pd.read_csv(f'{DATA}/{name}_test.csv')
        train_props = train[TARGET_COL].value_counts(normalize=True).sort_index()
        test_props = test[TARGET_COL].value_counts(normalize=True).sort_index()
        diff = (train_props - test_props).abs().max()
        assert diff < 0.03, f"{name}: class proportions differ by {diff:.3f}"


def test_smote_balanced():
    for name in ('admission', 'semester'):
        df = pd.read_csv(f'{DATA}/{name}_train_smote.csv')
        counts = df[TARGET_COL].value_counts()
        assert counts.nunique() == 1, f"{name}: classes not balanced — {counts.to_dict()}"


def test_smote_only_on_train():
    for name in ('admission', 'semester'):
        train = pd.read_csv(f'{DATA}/{name}_train.csv')
        smote = pd.read_csv(f'{DATA}/{name}_train_smote.csv')
        assert len(smote) > len(train)


def test_transform_new_student_returns_one_row():
    sample = {
        'Marital status': 'Single',
        'Gender': 'Male',
        'Scholarship holder': 'No',
        'Tuition fees up to date': 'Yes',
        'Age at enrollment': 20,
        'Admission grade': 120.5,
        'Previous qualification (grade)': 130.0,
    }
    df = transform_new_student(sample, stage='admission')
    assert df.shape[0] == 1, f"Expected 1 row, got {df.shape[0]}"


def test_transform_new_student_columns_match():
    sample = {'Marital status': 'Single', 'Gender': 'Male'}
    df = transform_new_student(sample, stage='admission')
    train = pd.read_csv(f'{DATA}/admission_train.csv')
    expected_cols = [c for c in train.columns if c != TARGET_COL]
    assert list(df.columns) == expected_cols


def test_transform_invalid_stage():
    with pytest.raises(ValueError):
        transform_new_student({}, stage='invalid')