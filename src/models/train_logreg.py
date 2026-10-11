"""
Baseline Logistic Regression for Student Dropout prediction.
Trains two models: admission-stage and semester-stage.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


RANDOM_STATE = 42
TARGET_COL = 'Target_encoded'
MODEL_DIR = 'models/production'


def load_data(stage):
    """Load train (SMOTE) and test sets for a stage."""
    train = pd.read_csv(f'data/processed/{stage}_train_smote.csv')
    test = pd.read_csv(f'data/processed/{stage}_test.csv')
    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]
    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    """Scale + fit Logistic Regression."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        multi_class='multinomial',
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def evaluate(model, scaler, X_test, y_test, stage):
    """Report metrics on test set."""
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

    # Recall for Dropout class (encoded 0)
    dropout_idx = list(model.classes_).index(0)
    dropout_recall = np.sum((y_pred == 0) & (y_test == 0)) / np.sum(y_test == 0)

    print(f"\n{'='*60}")
    print(f"MODEL: Logistic Regression — {stage.upper()}")
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


def save_artifacts(model, scaler, stage, metrics):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f'{MODEL_DIR}/logreg_{stage}.pkl')
    joblib.dump(scaler, f'{MODEL_DIR}/logreg_{stage}_scaler.pkl')
    print(f"\n  [SAVED] {MODEL_DIR}/logreg_{stage}.pkl")
    print(f"  [SAVED] {MODEL_DIR}/logreg_{stage}_scaler.pkl")
    return metrics


def train_stage(stage):
    print(f"\n>>> Loading {stage} data...")
    X_train, y_train, X_test, y_test = load_data(stage)
    print(f"    Train: {X_train.shape}  Test: {X_test.shape}")

    print(f">>> Training Logistic Regression ({stage})...")
    model, scaler = train_model(X_train, y_train)

    metrics = evaluate(model, scaler, X_test, y_test, stage)
    save_artifacts(model, scaler, stage, metrics)
    return metrics


def main():
    print("=" * 60)
    print("BASELINE MODEL: LOGISTIC REGRESSION")
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

    print("\n[DONE] Logistic Regression training complete.")


if __name__ == "__main__":
    main()