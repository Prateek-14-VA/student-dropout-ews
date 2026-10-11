"""
Per-student SHAP explanations.
For a given student row, computes:
  - Predicted class + probabilities
  - Top 5 factors pushing toward/away from Dropout
  - Human-readable explanations
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


TARGET_COL = 'Target_encoded'
MODEL_DIR = 'models/production'
CLASS_NAMES = ['Dropout', 'Enrolled', 'Graduate']


def load_model_and_data(stage):
    model = joblib.load(f'{MODEL_DIR}/production_{stage}.pkl')
    clf = model.named_steps['clf'] if hasattr(model, 'named_steps') else model
    test = pd.read_csv(f'data/processed/{stage}_test.csv')
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]
    return clf, X_test, y_test


def explain_student(clf, X_test, student_idx, top_n=5):
    """
    Compute SHAP values and produce human-readable explanation
    for a single student at a given index in X_test.
    """
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)

    # Normalize SHAP structure to list of 2D arrays per class
    if isinstance(shap_values, list):
        pass
    elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    student_row = X_test.iloc[student_idx]
    probs = clf.predict_proba(X_test.iloc[[student_idx]])[0]
    pred_class = int(np.argmax(probs))

    # SHAP values for Dropout class (index 0)
    dropout_shap = shap_values[0][student_idx]

    # Pair each feature with its SHAP value
    feature_names = X_test.columns.tolist()
    pairs = list(zip(feature_names, dropout_shap, student_row.values))

    # Sort by absolute value — most influential first
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'='*70}")
    print(f"STUDENT #{student_idx} EXPLANATION")
    print(f"{'='*70}")
    print(f"\nPredicted class:  {CLASS_NAMES[pred_class]}")
    print(f"Probabilities:")
    for i, cls in enumerate(CLASS_NAMES):
        marker = ' <--' if i == pred_class else ''
        print(f"  {cls:10s}: {probs[i]:.4f}{marker}")

    risk_score = probs[0]  # Dropout probability
    print(f"\nDropout Risk Score: {risk_score:.4f} ({risk_score*100:.1f}%)")

    print(f"\nTop {top_n} contributing features (SHAP for Dropout class):")
    print(f"{'-'*70}")
    print(f"{'Feature':45s} {'SHAP':>10s}  {'Effect':>10s}")
    print(f"{'-'*70}")
    for name, sv, val in pairs[:top_n]:
        direction = 'INCREASES' if sv > 0 else 'DECREASES'
        print(f"{name:45s} {sv:+10.4f}  {direction}")

    return {
        'student_idx': student_idx,
        'predicted_class': CLASS_NAMES[pred_class],
        'probabilities': dict(zip(CLASS_NAMES, probs.tolist())),
        'risk_score': float(risk_score),
        'top_factors': [
            {'feature': name, 'shap_value': float(sv), 'value': float(val)}
            for name, sv, val in pairs[:top_n]
        ]
    }


def main():
    print("=" * 70)
    print("PER-STUDENT SHAP EXPLANATIONS")
    print("=" * 70)

    for stage in ('admission', 'semester'):
        print(f"\n\n{'#'*70}")
        print(f"# {stage.upper()} STAGE — Explaining 3 sample students")
        print(f"{'#'*70}")

        clf, X_test, y_test = load_model_and_data(stage)
        print(f"  Test size: {len(X_test)}")

        # Explain 3 students: 0 (index), 100, 200
        for idx in [0, 100, 200]:
            actual = CLASS_NAMES[int(y_test.iloc[idx])]
            print(f"\n  [Actual label for student #{idx}: {actual}]")
            explain_student(clf, X_test, idx, top_n=5)

    print("\n" + "=" * 70)
    print("[DONE] Per-student explanations complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()