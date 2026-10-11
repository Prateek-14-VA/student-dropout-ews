"""
Fairness audit for production models.
Computes recall, FPR, and FNR for Dropout class across subgroups.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


TARGET_COL = 'Target_encoded'
MODEL_DIR = 'models/production'
ENCODER_DIR = 'models/encoders'
FIG_DIR = 'reports/figures'
REPORT_DIR = 'reports'
CLASS_NAMES = ['Dropout', 'Enrolled', 'Graduate']


def load_stage(stage):
    test = pd.read_csv(f'data/processed/{stage}_test.csv')
    X_test = test.drop(columns=[TARGET_COL]).copy()
    y_test = test[TARGET_COL].copy()

    model = joblib.load(f'{MODEL_DIR}/production_{stage}.pkl')
    clf = model.named_steps['clf'] if hasattr(model, 'named_steps') else model

    enc_path = f'{ENCODER_DIR}/{stage}_encoders.pkl'
    encoders = joblib.load(enc_path) if os.path.exists(enc_path) else {}

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    return X_test, y_test, y_pred, y_prob, encoders


def decode_column(X, encoders, col):
    if col not in encoders:
        return X[col].astype(str)
    le = encoders[col]
    return pd.Series(le.inverse_transform(X[col].astype(int)), index=X.index)


def compute_metrics(y_true, y_pred, mask):
    y_true_sub = y_true[mask]
    y_pred_sub = y_pred[mask]

    tp = np.sum((y_pred_sub == 0) & (y_true_sub == 0))
    fn = np.sum((y_pred_sub != 0) & (y_true_sub == 0))
    fp = np.sum((y_pred_sub == 0) & (y_true_sub != 0))
    tn = np.sum((y_pred_sub != 0) & (y_true_sub != 0))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        'n': int(len(y_true_sub)),
        'n_dropout': int((y_true_sub == 0).sum()),
        'recall_dropout': round(float(recall), 4),
        'false_positive_rate': round(float(fpr), 4),
        'false_negative_rate': round(float(fnr), 4),
    }


def audit_group(X_test, y_test, y_pred, encoders, sensitive_col, stage):
    print(f"\n  >>> Auditing: {sensitive_col}")
    values = decode_column(X_test, encoders, sensitive_col)
    rows = []

    for group_val in sorted(values.unique(), key=lambda x: str(x)):
        mask = (values == group_val).values
        if mask.sum() < 10:
            continue
        metrics = compute_metrics(y_test, y_pred, mask)
        metrics['group'] = str(group_val)
        metrics['sensitive_column'] = sensitive_col
        metrics['stage'] = stage
        rows.append(metrics)
        print(f"     {str(group_val):20s} n={metrics['n']:4d} "
              f"recall={metrics['recall_dropout']:.3f} "
              f"FPR={metrics['false_positive_rate']:.3f}")

    return rows


def compute_disparity(rows):
    if not rows:
        return 0.0
    recalls = [r['recall_dropout'] for r in rows]
    return round(max(recalls) - min(recalls), 4)


def plot_fairness(df, sensitive_col, stage):
    sub = df[(df['sensitive_column'] == sensitive_col) & (df['stage'] == stage)]
    if sub.empty:
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sub))
    width = 0.35

    ax.bar(x - width/2, sub['recall_dropout'], width, label='Recall', color='#27ae60')
    ax.bar(x + width/2, sub['false_positive_rate'], width, label='FPR', color='#e74c3c')

    ax.set_xticks(x)
    ax.set_xticklabels(sub['group'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Rate')
    ax.set_title(f'Fairness - {sensitive_col} - {stage.capitalize()} Stage')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    safe_name = sensitive_col.lower().replace(' ', '_').replace('/', '_')
    path = f'{FIG_DIR}/fairness_{stage}_{safe_name}.png'
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"     [SAVED] {path}")


def audit_stage(stage):
    print(f"\n{'='*60}")
    print(f"FAIRNESS AUDIT - {stage.upper()} STAGE")
    print(f"{'='*60}")

    X_test, y_test, y_pred, y_prob, encoders = load_stage(stage)
    print(f"  Test size: {len(X_test)}")

    if stage == 'admission':
        sensitive_cols = ['Gender', 'Scholarship holder', 'International', 'Debtor']
    else:
        sensitive_cols = ['Scholarship holder', 'Debtor', 'International']

    all_rows = []
    for col in sensitive_cols:
        if col not in X_test.columns:
            print(f"  [SKIP] Column not in {stage}: {col}")
            continue
        rows = audit_group(X_test, y_test, y_pred, encoders, col, stage)
        if rows:
            disparity = compute_disparity(rows)
            print(f"     Disparity (max-min recall) = {disparity:.4f}")
            all_rows.extend(rows)

    return all_rows


def main():
    print("=" * 60)
    print("FAIRNESS AUDIT")
    print("=" * 60)

    all_rows = []
    for stage in ('admission', 'semester'):
        all_rows.extend(audit_stage(stage))

    df = pd.DataFrame(all_rows)
    os.makedirs(REPORT_DIR, exist_ok=True)
    csv_path = f'{REPORT_DIR}/fairness_report.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n  [SAVED] {csv_path}")

    for stage in ('admission', 'semester'):
        for col in df['sensitive_column'].unique():
            plot_fairness(df, col, stage)

    print("\n" + "=" * 60)
    print("FAIRNESS SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))

    print("\n[DONE] Fairness audit complete.")


if __name__ == "__main__":
    main()