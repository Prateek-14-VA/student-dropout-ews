"""
Select production models and generate final evaluation artifacts.
Production models:
  - Admission stage: RandomForest (tuned)
  - Semester stage:  RandomForest (tuned)
"""
import os
import sys
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


TARGET_COL = 'Target_encoded'
MODEL_DIR = 'models/production'
FIG_DIR = 'reports/figures'
REPORT_DIR = 'reports'
CLASS_NAMES = ['Dropout', 'Enrolled', 'Graduate']


def load_test(stage):
    test = pd.read_csv(f'data/processed/{stage}_test.csv')
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]
    return X_test, y_test


def load_model(stage):
    path = f'{MODEL_DIR}/rf_tuned_{stage}.pkl'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}. Run tune.py first.")
    return joblib.load(path)


def evaluate_model(model, X_test, y_test, stage):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    dropout_recall = np.sum((y_pred == 0) & (y_test == 0)) / np.sum(y_test == 0)

    print(f"\n{'='*60}")
    print(f"FINAL MODEL - {stage.upper()} STAGE (RandomForest tuned)")
    print(f"{'='*60}")
    print(f"  Accuracy:        {acc:.4f}")
    print(f"  Macro F1:        {f1_macro:.4f}")
    print(f"  ROC-AUC (OvR):   {roc_auc:.4f}")
    print(f"  Dropout Recall:  {dropout_recall:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        'stage': stage,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'roc_auc': roc_auc,
        'dropout_recall': dropout_recall,
    }, y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred, stage):
    os.makedirs(FIG_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {stage.capitalize()}')
    for i in range(3):
        for j in range(3):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, cm[i, j], ha='center', va='center', color=color)
    plt.colorbar(im)
    plt.tight_layout()
    path = f'{FIG_DIR}/confusion_matrix_{stage}.png'
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  [SAVED] {path}")


def plot_roc_curves(y_test, y_prob, stage):
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(CLASS_NAMES):
        y_bin = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        auc_i = roc_auc_score(y_bin, y_prob[:, i])
        plt.plot(fpr, tpr, label=f'{cls} (AUC = {auc_i:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {stage.capitalize()} Stage')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = f'{FIG_DIR}/roc_curves_{stage}.png'
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  [SAVED] {path}")


def copy_production(stage):
    src = f'{MODEL_DIR}/rf_tuned_{stage}.pkl'
    dst = f'{MODEL_DIR}/production_{stage}.pkl'
    shutil.copy(src, dst)
    print(f"  [PRODUCTION] {dst}")


def main():
    print("=" * 60)
    print("FINAL MODEL SELECTION")
    print("=" * 60)

    all_metrics = []
    for stage in ('admission', 'semester'):
        print(f"\n>>> Loading {stage} test set + tuned RF...")
        X_test, y_test = load_test(stage)
        model = load_model(stage)

        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, stage)
        all_metrics.append(metrics)

        plot_confusion_matrix(y_test, y_pred, stage)
        plot_roc_curves(y_test, y_prob, stage)
        copy_production(stage)

    df = pd.DataFrame(all_metrics)
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = f'{REPORT_DIR}/final_metrics.csv'
    df.to_csv(path, index=False)
    print(f"\n  [SAVED] {path}")

    print("\n" + "=" * 60)
    print("FINAL METRICS SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))

    print("\n[DONE] Final model selection complete.")
    print("Production models:")
    print(f"  - {MODEL_DIR}/production_admission.pkl")
    print(f"  - {MODEL_DIR}/production_semester.pkl")


if __name__ == "__main__":
    main()