"""
Cross-validation evaluation for all 6 baseline models.
Categorical columns are label-encoded before CV.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


RANDOM_STATE = 42
TARGET_COL = 'Target_encoded'
N_SPLITS = 5
FIG_DIR = 'reports/figures'
REPORT_DIR = 'reports'


def load_full_data(stage):
    """Load features CSV and label-encode object columns."""
    df = pd.read_csv(f'data/processed/{stage}_features.csv')
    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL]).copy()

    # Label-encode any remaining object columns
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    return X, y


def get_models():
    return {
        'LogReg': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000, multi_class='multinomial',
                class_weight='balanced', random_state=RANDOM_STATE))
        ]),
        'RandomForest': RandomForestClassifier(
            n_estimators=300, min_samples_split=5, min_samples_leaf=2,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective='multi:softprob', num_class=3, eval_metric='mlogloss',
            random_state=RANDOM_STATE, n_jobs=-1, tree_method='hist'),
    }


def cross_validate_stage(stage):
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION - {stage.upper()} STAGE")
    print(f"{'='*60}")

    X, y = load_full_data(stage)
    print(f"  Data: {X.shape}  Classes: {y.value_counts().sort_index().to_dict()}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'roc_auc_ovr': 'roc_auc_ovr',
    }

    results = []
    for name, model in get_models().items():
        print(f"\n  >>> {name}")
        row = {'stage': stage, 'model': name}
        for metric_name, scorer in scoring.items():
            scores = cross_val_score(model, X, y, cv=cv,
                                     scoring=scorer, n_jobs=-1)
            row[f'{metric_name}_mean'] = scores.mean()
            row[f'{metric_name}_std'] = scores.std()
            print(f"     {metric_name:15s}: {scores.mean():.4f} +/- {scores.std():.4f}")
        results.append(row)

    return pd.DataFrame(results)


def plot_comparison(df):
    os.makedirs(FIG_DIR, exist_ok=True)
    metrics = ['accuracy_mean', 'f1_macro_mean', 'roc_auc_ovr_mean']
    titles = ['Accuracy', 'Macro F1', 'ROC-AUC (OvR)']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, title in zip(axes, metrics, titles):
        for stage, color in [('admission', '#3498db'), ('semester', '#e74c3c')]:
            sub = df[df['stage'] == stage]
            ax.bar(np.arange(len(sub)) + (0.35 if stage == 'semester' else 0),
                   sub[metric], width=0.35, label=stage.capitalize(), color=color)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(np.arange(3) + 0.175)
        ax.set_xticklabels(['LogReg', 'RF', 'XGB'])
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Model Comparison - 5-Fold CV', fontsize=14, y=1.02)
    plt.tight_layout()
    path = f'{FIG_DIR}/model_comparison.png'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n  [SAVED] {path}")


def main():
    print("=" * 60)
    print("CROSS-VALIDATION OF ALL BASELINE MODELS")
    print("=" * 60)

    dfs = []
    for stage in ('admission', 'semester'):
        dfs.append(cross_validate_stage(stage))

    df = pd.concat(dfs, ignore_index=True)

    os.makedirs(REPORT_DIR, exist_ok=True)
    csv_path = f'{REPORT_DIR}/model_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n  [SAVED] {csv_path}")

    plot_comparison(df)

    print("\n" + "=" * 60)
    print("FINAL COMPARISON TABLE")
    print("=" * 60)
    pivot = df.pivot_table(
        index='model', columns='stage',
        values=['accuracy_mean', 'f1_macro_mean', 'roc_auc_ovr_mean']
    ).round(4)
    print(pivot)

    print("\n[DONE] Cross-validation complete.")


if __name__ == "__main__":
    main()