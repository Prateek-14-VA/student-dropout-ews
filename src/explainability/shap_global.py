"""
SHAP Global Feature Importance for production models.
Generates:
  - Global feature importance plot (bar)
  - SHAP summary plot (beeswarm)
  - Saves raw SHAP values for later use
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


TARGET_COL = 'Target_encoded'
MODEL_DIR = 'models/production'
FIG_DIR = 'reports/figures'
SHAP_DIR = 'models/shap'
CLASS_NAMES = ['Dropout', 'Enrolled', 'Graduate']


def load_test(stage):
    test = pd.read_csv(f'data/processed/{stage}_test.csv')
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]
    return X_test, y_test


def load_model(stage):
    path = f'{MODEL_DIR}/production_{stage}.pkl'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    model = joblib.load(path)
    # Our production models are imblearn pipelines - extract the classifier
    if hasattr(model, 'named_steps'):
        clf = model.named_steps['clf']
    else:
        clf = model
    return clf


def compute_shap(clf, X_test, stage):
    print(f"\n>>> Computing SHAP values for {stage.upper()}...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)

    # For multiclass RF, shap_values may be list of arrays (one per class)
    # OR a 3D array depending on shap version. Normalize to list of 2D arrays.
    if isinstance(shap_values, list):
        pass
    elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    else:
        shap_values = [shap_values]

    print(f"    SHAP values: {len(shap_values)} classes")
    for i, sv in enumerate(shap_values):
        print(f"    Class {CLASS_NAMES[i]}: shape {sv.shape}")

    os.makedirs(SHAP_DIR, exist_ok=True)
    np.savez(f'{SHAP_DIR}/{stage}_shap_values.npz',
             **{CLASS_NAMES[i]: sv for i, sv in enumerate(shap_values)})
    print(f"    [SAVED] {SHAP_DIR}/{stage}_shap_values.npz")

    return shap_values


def plot_global_importance(shap_values, X_test, stage):
    """Bar chart of mean |SHAP| per feature - averaged across classes."""
    os.makedirs(FIG_DIR, exist_ok=True)

    mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)

    order = np.argsort(mean_abs)[::-1]
    top_n = min(20, len(order))
    idx = order[:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(np.arange(top_n),
            mean_abs[idx][::-1],
            color='#3498db')
    ax.set_yticks(np.arange(top_n))
    ax.set_yticklabels([X_test.columns[i] for i in idx][::-1])
    ax.set_xlabel('Mean |SHAP value| (average across classes)')
    ax.set_title(f'Global Feature Importance - {stage.capitalize()} Stage')
    plt.tight_layout()
    path = f'{FIG_DIR}/shap_global_{stage}.png'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    [SAVED] {path}")


def plot_summary_beeswarm(shap_values, X_test, stage):
    """SHAP summary plot per class."""
    os.makedirs(FIG_DIR, exist_ok=True)

    for i, cls in enumerate(CLASS_NAMES):
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[i], X_test,
                          plot_type='dot', show=False, max_display=15)
        plt.title(f'SHAP Summary - {cls} - {stage.capitalize()} Stage')
        plt.tight_layout()
        path = f'{FIG_DIR}/shap_summary_{stage}_{cls.lower()}.png'
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"    [SAVED] {path}")


def process_stage(stage):
    print(f"\n{'='*60}")
    print(f"SHAP GLOBAL - {stage.upper()} STAGE")
    print(f"{'='*60}")

    X_test, y_test = load_test(stage)
    clf = load_model(stage)
    print(f"  Test data: {X_test.shape}")
    print(f"  Classifier: {type(clf).__name__}")

    shap_values = compute_shap(clf, X_test, stage)
    plot_global_importance(shap_values, X_test, stage)
    plot_summary_beeswarm(shap_values, X_test, stage)


def main():
    print("=" * 60)
    print("SHAP GLOBAL FEATURE IMPORTANCE")
    print("=" * 60)

    for stage in ('admission', 'semester'):
        process_stage(stage)

    print("\n" + "=" * 60)
    print("[DONE] SHAP global analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()