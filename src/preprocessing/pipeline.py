"""
Reusable end-to-end preprocessing pipeline.
Runs: clean -> encode -> features -> split + SMOTE.
"""
import os
import sys
import subprocess
import pandas as pd
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


STEPS = [
    ('Clean',       'src/preprocessing/clean.py'),
    ('Encode',      'src/preprocessing/encode.py'),
    ('Features',    'src/preprocessing/features.py'),
    ('Split+SMOTE', 'src/preprocessing/split.py'),
]


def run_step(name, script):
    print("\n" + "=" * 60)
    print(f"STEP: {name}  ({script})")
    print("=" * 60)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {name}")


def run_full_pipeline():
    print("\n*** RUNNING FULL PREPROCESSING PIPELINE ***")
    for name, script in STEPS:
        run_step(name, script)
    print("\n" + "*" * 60)
    print("*** PIPELINE COMPLETE ***")
    print("*" * 60)


def transform_new_student(student_dict, stage='admission'):
    """Transform a single new student dict into a numeric feature row."""
    if stage not in ('admission', 'semester'):
        raise ValueError("stage must be 'admission' or 'semester'")

    enc_path = f'models/encoders/{stage}_encoders.pkl'
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Encoders not found: {enc_path}")
    encoders = joblib.load(enc_path)

    train_path = f'data/processed/{stage}_train.csv'
    cols = pd.read_csv(train_path, nrows=0).columns.tolist()
    cols = [c for c in cols if c != 'Target_encoded']

    row = {}
    for col in cols:
        val = student_dict.get(col, 0)
        if col in encoders:
            le = encoders[col]
            try:
                row[col] = int(val) if isinstance(val, (int, float)) else int(le.transform([str(val)])[0])
            except (ValueError, KeyError):
                row[col] = 0
        else:
            row[col] = val

    return pd.DataFrame([row])[cols]


def main():
    run_full_pipeline()

    print("\n[SMOKE TEST] transform_new_student()")
    sample = {
        'Marital status': 'Single',
        'Gender': 'Male',
        'Scholarship holder': 'No',
        'Tuition fees up to date': 'Yes',
        'Age at enrollment': 20,
        'Admission grade': 120.5,
        'Previous qualification (grade)': 130.0,
    }
    try:
        df = transform_new_student(sample, stage='admission')
        print(f"   Output shape: {df.shape}")
        print("   [OK] transform_new_student works")
    except Exception as e:
        print(f"   [WARN] Smoke test failed: {e}")


if __name__ == "__main__":
    main()