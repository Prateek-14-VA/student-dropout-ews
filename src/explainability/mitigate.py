"""
Bias mitigation via group-specific decision thresholds.
Goal: Equalize Dropout recall across groups (e.g., Female vs Male).

Approach:
  1. Get predicted Dropout probability for each student
  2. For each group, find the threshold that achieves target recall
  3. Reclassify with group thresholds
  4. Compare original vs mitigated fairness metrics
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


def load_stage(stage):
    test = pd.read_csv(f'data/processed/{stage}_test.csv')
    X_test = test.drop(columns=[TARGET_COL]).copy()
    y_test = test[TARGET_COL].copy()

    model = joblib.load(f'{MODEL_DIR}/production_{stage}.pkl')
    clf = model.named_steps['clf'] if hasattr(model, 'named_steps') else model

    encoders = joblib.load(f'{ENCODER_DIR}/{stage}_encoders.pkl')

    y_prob = clf.predict_proba(X_test)
    return X_test, y_test, y_prob, encoders


def decode_column(X, encoders, col):
    le = encoders[col]
    return pd.Series(le.inverse_transform(X[col].astype(int)), index=X.index)


def find_threshold_for_recall(y_true_dropout_mask, prob_dropout, target_recall=0.70):
    """
    Given probabilities, find the threshold that achieves target recall
    on the Dropout class for a specific group.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_gap = 1.0

    for t in thresholds:
        pred_dropout = prob_dropout >= t
        tp = np.sum(pred_dropout & y_true_dropout_mask)
        fn = np.sum(~pred_dropout & y_true_dropout_mask)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        gap = abs(recall - target_recall)
        if gap < best_gap:
            best_gap = gap
            best_threshold = t

    return best_threshold


def apply_group_thresholds(y_test, y_prob, groups, thresholds):
    """Reclassify using group-specific thresholds for Dropout."""
    y_pred_new = np.argmax(y_prob, axis=1).copy()
    for group_val, threshold in thresholds.items():
        mask = (groups == group_val).values
        # If Dropout probability >= threshold, force class 0 (Dropout)
        force_dropout = (y_prob[:, 0] >= threshold) & mask
        y_pred_new[force_dropout] = 0
    return y_pred_new


def compute_group_metrics(y_true, y_pred, mask):
    y_true_sub = y_true[mask]
    y_pred_sub = y_pred[mask]

    tp = np.sum((y_pred_sub == 0) & (y_true_sub == 0))
    fn = np.sum((y_pred_sub != 0) & (y_true_sub == 0))
    fp = np.sum((y_pred_sub == 0) & (y_true_sub != 0))
    tn = np.sum((y_pred_sub != 0) & (y_true_sub != 0))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {'recall': recall, 'fpr': fpr, 'n': int(mask.sum())}


def audit_fairness(y_test, y_pred, groups, label):
    """Return per-group recall + FPR."""
    rows = []
    for g in sorted(groups.unique(), key=lambda x: str(x)):
        mask = (groups == g).values
        if mask.sum() < 10:
            continue
        m = compute_group_metrics(y_test, y_pred, mask)
        m['group'] = str(g)
        m['label'] = label
        rows.append(m)
    return rows


def plot_comparison(before_rows, after_rows, sensitive_col, stage):
    df_b = pd.DataFrame(before_rows)
    df_a = pd.DataFrame(after_rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(df_b))
    width = 0.35

    # Recall
    axes[0].bar(x - width/2, df_b['recall'], width, label='Original', color='#e74c3c')
    axes[0].bar(x + width/2, df_a['recall'], width, label='Mitigated', color='#27ae60')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_b['group'])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Recall')
    axes[0].set_title(f'Dropout Recall - {sensitive_col}')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # FPR
    axes[1].bar(x - width/2, df_b['fpr'], width, label='Original', color='#e74c3c')
    axes[1].bar(x + width/2, df_a['fpr'], width, label='Mitigated', color='#27ae60')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_b['group'])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('FPR')
    axes[1].set_title(f'False Positive Rate - {sensitive_col}')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle(f'Bias Mitigation - {sensitive_col} - {stage.capitalize()} Stage', fontsize=13)
    plt.tight_layout()
    safe = sensitive_col.lower().replace(' ', '_')
    path = f'{FIG_DIR}/mitigation_{stage}_{safe}.png'
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"     [SAVED] {path}")


def mitigate_stage(stage, sensitive_col='Gender', target_recall=0.70):
    print(f"\n{'='*60}")
    print(f"MITIGATION - {stage.upper()} - {sensitive_col}")
    print(f"{'='*60}")

    X_test, y_test, y_prob, encoders = load_stage(stage)

    if sensitive_col not in X_test.columns:
        print(f"  [SKIP] {sensitive_col} not in {stage}")
        return None

    groups = decode_column(X_test, encoders, sensitive_col)
    y_true = y_test.values
    y_pred_orig = np.argmax(y_prob, axis=1)

    # Before
    before_rows = audit_fairness(y_test, y_pred_orig, groups, 'original')
    print(f"\n  BEFORE mitigation:")
    for r in before_rows:
        print(f"     {r['group']:15s} recall={r['recall']:.3f}  FPR={r['fpr']:.3f}")
    before_disp = max(r['recall'] for r in before_rows) - min(r['recall'] for r in before_rows)
    print(f"     Disparity (recall): {before_disp:.4f}")

    # Find optimal threshold per group
    thresholds = {}
    for g in sorted(groups.unique(), key=lambda x: str(x)):
        mask = (groups == g).values
        if mask.sum() < 10:
            continue
        y_true_g = (y_true[mask] == 0).astype(int)
        prob_g = y_prob[:, 0][mask]
        t = find_threshold_for_recall(y_true_g == 1, prob_g, target_recall=target_recall)
        thresholds[g] = t
    print(f"\n  Group-specific thresholds (target recall={target_recall}):")
    for g, t in thresholds.items():
        print(f"     {g:15s} threshold = {t:.2f}")

    # Apply thresholds
    y_pred_mit = apply_group_thresholds(y_test, y_prob, groups, thresholds)

    # After
    after_rows = audit_fairness(y_test, y_pred_mit, groups, 'mitigated')
    print(f"\n  AFTER mitigation:")
    for r in after_rows:
        print(f"     {r['group']:15s} recall={r['recall']:.3f}  FPR={r['fpr']:.3f}")
    after_disp = max(r['recall'] for r in after_rows) - min(r['recall'] for r in after_rows)
    print(f"     Disparity (recall): {after_disp:.4f}")
    print(f"\n  Disparity reduction: {before_disp - after_disp:+.4f}")

    plot_comparison(before_rows, after_rows, sensitive_col, stage)

    return {
        'stage': stage,
        'sensitive_col': sensitive_col,
        'before_disparity': before_disp,
        'after_disparity': after_disp,
        'thresholds': thresholds,
        'before_rows': before_rows,
        'after_rows': after_rows,
    }


def main():
    print("=" * 60)
    print("BIAS MITIGATION - GROUP-SPECIFIC THRESHOLDS")
    print("=" * 60)

    results = []
    # Gender only exists in admission
    r = mitigate_stage('admission', sensitive_col='Gender', target_recall=0.70)
    if r:
        results.append(r)

    # Scholarship exists in both
    for stage in ('admission', 'semester'):
        r = mitigate_stage(stage, sensitive_col='Scholarship holder', target_recall=0.65)
        if r:
            results.append(r)

    # Save summary CSV
    summary = []
    for r in results:
        summary.append({
            'stage': r['stage'],
            'attribute': r['sensitive_col'],
            'before_disparity': r['before_disparity'],
            'after_disparity': r['after_disparity'],
            'improvement': r['before_disparity'] - r['after_disparity'],
        })
    df = pd.DataFrame(summary)
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = f'{REPORT_DIR}/mitigation_report.csv'
    df.to_csv(path, index=False)
    print(f"\n  [SAVED] {path}")

    print("\n" + "=" * 60)
    print("MITIGATION SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))

    print("\n[DONE] Bias mitigation complete.")


if __name__ == "__main__":
    main()