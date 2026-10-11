"""
Hyperparameter tuning with SMOTE applied INSIDE CV folds.
Uses imblearn Pipeline to prevent data leakage.
Test set is never touched.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


RANDOM_STATE = 42
TARGET_COL = 'Target_encoded'
N_ITER = 20
CV_FOLDS = 3
MODEL_DIR = 'models/production'


def load_train_only(stage):
    """
    Load RAW train split (NO SMOTE — SMOTE will be applied inside CV folds).
    """
    train = pd.read_csv(f'data/processed/{stage}_train.csv')
    y = train[TARGET_COL].copy()
    X = train.drop(columns=[TARGET_COL]).copy()
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X, y


def rf_search_space():
    return {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [None, 10, 15, 20],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['sqrt', 'log2'],
    }


def xgb_search_space():
    return {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [3, 4, 6, 8],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__subsample': [0.6, 0.8, 1.0],
        'clf__colsample_bytree': [0.6, 0.8, 1.0],
    }


def build_rf_pipeline():
    return ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf', RandomForestClassifier(
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])


def build_xgb_pipeline():
    return ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf', XGBClassifier(
            objective='multi:softprob', num_class=3, eval_metric='mlogloss',
            random_state=RANDOM_STATE, n_jobs=-1, tree_method='hist'
        ))
    ])


def tune_rf(X, y):
    print("\n>>> Tuning RandomForest (SMOTE inside CV)...")
    search = RandomizedSearchCV(
        build_rf_pipeline(), rf_search_space(),
        n_iter=N_ITER,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring='f1_macro', n_jobs=-1, random_state=RANDOM_STATE, verbose=1
    )
    search.fit(X, y)
    print(f"    Best CV F1: {search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_xgb(X, y):
    print("\n>>> Tuning XGBoost (SMOTE inside CV)...")
    search = RandomizedSearchCV(
        build_xgb_pipeline(), xgb_search_space(),
        n_iter=N_ITER,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring='f1_macro', n_jobs=-1, random_state=RANDOM_STATE, verbose=1
    )
    search.fit(X, y)
    print(f"    Best CV F1: {search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_stage(stage):
    print(f"\n{'='*60}")
    print(f"TUNING - {stage.upper()} STAGE (SMOTE inside CV)")
    print(f"{'='*60}")

    X, y = load_train_only(stage)
    print(f"  Raw train data: {X.shape}")
    print(f"  Class dist: {y.value_counts().sort_index().to_dict()}")

    results = {}

    rf_model, rf_params, rf_score = tune_rf(X, y)
    joblib.dump(rf_model, f'{MODEL_DIR}/rf_tuned_{stage}.pkl')
    print(f"    [SAVED] {MODEL_DIR}/rf_tuned_{stage}.pkl")
    results['RandomForest'] = {'best_score': rf_score, 'best_params': rf_params}

    xgb_model, xgb_params, xgb_score = tune_xgb(X, y)
    joblib.dump(xgb_model, f'{MODEL_DIR}/xgb_tuned_{stage}.pkl')
    print(f"    [SAVED] {MODEL_DIR}/xgb_tuned_{stage}.pkl")
    results['XGBoost'] = {'best_score': xgb_score, 'best_params': xgb_params}

    return results


def main():
    print("=" * 60)
    print("HYPERPARAMETER TUNING (SMOTE inside CV — no leakage)")
    print("=" * 60)

    all_results = {}
    for stage in ('admission', 'semester'):
        all_results[stage] = tune_stage(stage)

    os.makedirs('reports', exist_ok=True)
    with open('reports/tuning_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  [SAVED] reports/tuning_results.json")

    print("\n" + "=" * 60)
    print("TUNING SUMMARY")
    print("=" * 60)
    for stage, models in all_results.items():
        print(f"\n  {stage.upper()}:")
        for name, info in models.items():
            print(f"    {name:15s}  best CV F1 = {info['best_score']:.4f}")

    print("\n[DONE] Hyperparameter tuning complete.")


if __name__ == "__main__":
    main()