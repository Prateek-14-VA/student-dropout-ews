"""
Baseline XGBoost for Student Dropout prediction.
Trains two models: admission-stage and semester-stage.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


RANDOM_STATE = 42
TARGET_COL = 'Target_encoded'
MODEL_DIR = 'models/production'


def load_data(stage):
    train = pd.read_csv(f'data/processed/{stage}_train_smote.csv')
    test = pd.read_csv(f'data/processed/{stage}_test.csv')
    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]
    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method='hist'
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, stage):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    dropout_recall = np.sum((y_pred == 0) & (y_test == 0)) / np.sum(y_test == 0)

    print(f"\n{'='*60}")
    print(f"MODEL: XGBoost - {stage.upper()}")
    print(f"{'='*60}")
    print(f"  Accuracy:        {acc:.4f}")
    print(f"  Macro F1:        {f1_macro:.4f}")
    print(f"  ROC-AUC (OvR):   {roc_auc:.4f}")
    print(f"  Dropout Recall:  {dropout_recall:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Dropout', 'Enrolled', 'Graduate']))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {'accuracy': acc, 'f1_macro': f1_macro,
            'roc_auc': roc_auc, 'dropout_recall': dropout_recall}


def save_artifacts(model, stage):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f'{MODEL_DIR}/xgb_{stage}.pkl')
    print(f"\n  [SAVED] {MODEL_DIR}/xgb_{stage}.pkl")


def print_feature_importance(model, feature_names, top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    print(f"\n  Top {top_n} features:")
    for i, idx in enumerate(indices, 1):
        print(f"    {i:2d}. {feature_names[idx]:45s} {importances[idx]:.4f}")


def train_stage(stage):
    print(f"\n>>> Loading {stage} data...")
    X_train, y_train, X_test, y_test = load_data(stage)
    print(f"    Train: {X_train.shape}  Test: {X_test.shape}")

    print(f">>> Training XGBoost ({stage})...")
    model = train_model(X_train, y_train)

    metrics = evaluate(model, X_test, y_test, stage)
    save_artifacts(model, stage)
    print_feature_importance(model, list(X_train.columns))
    return metrics


def main():
    print("=" * 60)
    print("BASELINE MODEL: XGBOOST")
    print("=" * 60)

    results = {}
    for stage in ('admission', 'semester'):
        results[stage] = train_stage(stage)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for stage, m in results.items():
        print(f"\n  {stage.upper()}:")
        for k, v in m.items():
            print(f"    {k:20s} = {v:.4f}")

    print("\n[DONE] XGBoost training complete.")


if __name__ == "__main__":
    main()