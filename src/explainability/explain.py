"""
API-ready student explainer.
Takes a raw student dict (human-readable) and returns JSON-friendly output.
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
ENCODER_DIR = 'models/encoders'
CLASS_NAMES = ['Dropout', 'Enrolled', 'Graduate']


INTERVENTION_RULES = {
    'Tuition fees up to date': {
        'suggestion': 'Financial counselling / fee instalment plan',
        'category': 'financial',
    },
    'Debtor': {
        'suggestion': 'Financial aid review - check eligibility for scholarships',
        'category': 'financial',
    },
    'Scholarship holder': {
        'suggestion': 'Scholarship guidance - help apply for government schemes',
        'category': 'financial',
    },
    'Curricular units 1st sem (approved)': {
        'suggestion': 'Academic mentoring - review first semester performance',
        'category': 'academic',
    },
    'Curricular units 2nd sem (approved)': {
        'suggestion': 'Academic mentoring - second semester grade support',
        'category': 'academic',
    },
    'Curricular units 1st sem (grade)': {
        'suggestion': 'Subject-specific tutoring for weak areas',
        'category': 'academic',
    },
    'Curricular units 2nd sem (grade)': {
        'suggestion': 'Subject-specific tutoring for weak areas',
        'category': 'academic',
    },
    'Age at enrollment': {
        'suggestion': 'Personal counselling - mature student support group',
        'category': 'personal',
    },
}


def _load_model(stage):
    path = f'{MODEL_DIR}/production_{stage}.pkl'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    clf = model.named_steps['clf'] if hasattr(model, 'named_steps') else model
    return clf


def _load_encoders(stage):
    path = f'{ENCODER_DIR}/{stage}_encoders.pkl'
    if os.path.exists(path):
        return joblib.load(path)
    return {}


def _load_feature_order(stage):
    train = pd.read_csv(f'data/processed/{stage}_train.csv', nrows=0)
    return [c for c in train.columns if c != TARGET_COL]


def _to_feature_row(student_input, stage):
    encoders = _load_encoders(stage)
    feature_cols = _load_feature_order(stage)

    row = {}
    for col in feature_cols:
        val = student_input.get(col, 0)

        if col in encoders and isinstance(val, str):
            le = encoders[col]
            try:
                val = int(le.transform([val])[0])
            except ValueError:
                val = int(le.transform([le.classes_[0]])[0])
        row[col] = val

    return pd.DataFrame([row])[feature_cols]


def _intervention_for(feature_name, shap_value):
    if shap_value <= 0:
        return None
    rule = INTERVENTION_RULES.get(feature_name)
    if not rule:
        return None
    return {
        'feature': feature_name,
        'category': rule['category'],
        'suggestion': rule['suggestion'],
    }


def explain_student(student_input, stage='admission', top_n=5):
    """Main API entry point."""
    if stage not in ('admission', 'semester'):
        raise ValueError("stage must be 'admission' or 'semester'")

    clf = _load_model(stage)

    if isinstance(student_input, pd.DataFrame):
        X_row = student_input
    else:
        X_row = _to_feature_row(student_input, stage)

    probs = clf.predict_proba(X_row)[0]
    pred_class_idx = int(np.argmax(probs))

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_row)

    if isinstance(sv, list):
        shap_row = sv[0][0]
    elif hasattr(sv, 'shape') and len(sv.shape) == 3:
        shap_row = sv[0, :, 0]
    else:
        shap_row = sv[0]

    feature_names = X_row.columns.tolist()
    raw_values = X_row.iloc[0].values

    pairs = sorted(zip(feature_names, shap_row, raw_values),
                   key=lambda x: abs(x[1]), reverse=True)[:top_n]

    top_factors = []
    for name, sv_val, raw in pairs:
        top_factors.append({
            'feature': name,
            'shap_value': round(float(sv_val), 4),
            'impact': 'increases risk' if sv_val > 0 else 'decreases risk',
            'value': float(raw),
        })

    interventions = []
    for f in top_factors:
        sug = _intervention_for(f['feature'], f['shap_value'])
        if sug:
            interventions.append(sug)

    return {
        'stage': stage,
        'predicted_class': CLASS_NAMES[pred_class_idx],
        'risk_score': round(float(probs[0]), 4),
        'probabilities': {cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, probs)},
        'top_factors': top_factors,
        'suggested_interventions': interventions,
    }


def main():
    print("=" * 70)
    print("explain_student() SMOKE TEST")
    print("=" * 70)

    low_risk = {
        'Marital status': 'Single',
        'Gender': 'Male',
        'Scholarship holder': 'Yes',
        'Tuition fees up to date': 'Yes',
        'Age at enrollment': 19,
        'Admission grade': 140.0,
        'Previous qualification (grade)': 145.0,
        'Debtor': 'No',
    }

    high_risk = {
        'Marital status': 'Married',
        'Gender': 'Female',
        'Scholarship holder': 'No',
        'Tuition fees up to date': 'No',
        'Age at enrollment': 32,
        'Admission grade': 105.0,
        'Previous qualification (grade)': 110.0,
        'Debtor': 'Yes',
    }

    for label, student in [('LOW-RISK', low_risk), ('HIGH-RISK', high_risk)]:
        print(f"\n{'='*70}")
        print(f"{label} STUDENT - Admission Stage")
        print(f"{'='*70}")
        result = explain_student(student, stage='admission', top_n=5)

        print(f"  Predicted: {result['predicted_class']}")
        print(f"  Risk score: {result['risk_score']:.4f}")
        print(f"  Probabilities: {result['probabilities']}")
        print(f"\n  Top factors:")
        for f in result['top_factors']:
            print(f"    {f['feature']:45s} SHAP={f['shap_value']:+.4f}  ({f['impact']})")
        print(f"\n  Suggested interventions:")
        if result['suggested_interventions']:
            for s in result['suggested_interventions']:
                print(f"    [{s['category']}] {s['suggestion']}")
        else:
            print(f"    None (student is low-risk)")

    print("\n" + "=" * 70)
    print("[DONE] Smoke test complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()