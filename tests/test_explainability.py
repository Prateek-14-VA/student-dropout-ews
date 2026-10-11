"""
Unit tests for the explainability module.
Run with: pytest tests/test_explainability.py -v
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.explainability.explain import explain_student


# ---------- Test data ----------

LOW_RISK_STUDENT = {
    'Marital status': 'Single',
    'Gender': 'Male',
    'Scholarship holder': 'Yes',
    'Tuition fees up to date': 'Yes',
    'Age at enrollment': 19,
    'Admission grade': 140.0,
    'Previous qualification (grade)': 145.0,
    'Debtor': 'No',
}

HIGH_RISK_STUDENT = {
    'Marital status': 'Married',
    'Gender': 'Female',
    'Scholarship holder': 'No',
    'Tuition fees up to date': 'No',
    'Age at enrollment': 32,
    'Admission grade': 105.0,
    'Previous qualification (grade)': 110.0,
    'Debtor': 'Yes',
}


# ---------- Basic return structure ----------

def test_explain_returns_dict():
    result = explain_student(LOW_RISK_STUDENT, stage='admission')
    assert isinstance(result, dict)


def test_explain_required_keys():
    result = explain_student(LOW_RISK_STUDENT, stage='admission')
    required = {'stage', 'predicted_class', 'risk_score', 'probabilities',
                'top_factors', 'suggested_interventions'}
    assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"


def test_predicted_class_is_valid():
    result = explain_student(LOW_RISK_STUDENT, stage='admission')
    assert result['predicted_class'] in ('Dropout', 'Enrolled', 'Graduate')


def test_risk_score_in_range():
    result = explain_student(HIGH_RISK_STUDENT, stage='admission')
    assert 0.0 <= result['risk_score'] <= 1.0


def test_probabilities_sum_to_one():
    result = explain_student(LOW_RISK_STUDENT, stage='admission')
    total = sum(result['probabilities'].values())
    assert abs(total - 1.0) < 0.01, f"Probabilities sum = {total}"


def test_probabilities_has_three_classes():
    result = explain_student(LOW_RISK_STUDENT, stage='admission')
    assert set(result['probabilities'].keys()) == {'Dropout', 'Enrolled', 'Graduate'}


# ---------- Top factors ----------

def test_top_factors_count():
    result = explain_student(LOW_RISK_STUDENT, stage='admission', top_n=5)
    assert len(result['top_factors']) == 5


def test_top_factors_have_required_fields():
    result = explain_student(LOW_RISK_STUDENT, stage='admission', top_n=3)
    for f in result['top_factors']:
        assert 'feature' in f
        assert 'shap_value' in f
        assert 'impact' in f
        assert f['impact'] in ('increases risk', 'decreases risk')


def test_top_factors_sorted_by_absolute_shap():
    result = explain_student(HIGH_RISK_STUDENT, stage='admission', top_n=5)
    shap_abs = [abs(f['shap_value']) for f in result['top_factors']]
    # Should be in descending order
    for i in range(len(shap_abs) - 1):
        assert shap_abs[i] >= shap_abs[i+1] - 0.0001


# ---------- High risk vs low risk ----------

def test_high_risk_has_higher_risk_score():
    low = explain_student(LOW_RISK_STUDENT, stage='admission')
    high = explain_student(HIGH_RISK_STUDENT, stage='admission')
    assert high['risk_score'] > low['risk_score'], \
        f"High risk {high['risk_score']} should exceed low {low['risk_score']}"


# ---------- Interventions ----------

def test_interventions_are_list():
    result = explain_student(HIGH_RISK_STUDENT, stage='admission')
    assert isinstance(result['suggested_interventions'], list)


def test_high_risk_student_has_interventions():
    result = explain_student(HIGH_RISK_STUDENT, stage='admission')
    # High-risk student should have at least one suggested intervention
    # (fee not paid, debtor, etc.)
    assert len(result['suggested_interventions']) > 0


# ---------- Stage parameter ----------

def test_invalid_stage_raises():
    with pytest.raises(ValueError):
        explain_student(LOW_RISK_STUDENT, stage='invalid')


def test_semester_stage_works():
    # Semester features differ, but function should accept the call
    semester_student = {
        'Marital status': 'Single',
        'Gender': 'Male',
        'Scholarship holder': 'No',
        'Tuition fees up to date': 'Yes',
        'Age at enrollment': 20,
        'Admission grade': 125.0,
        'Curricular units 1st sem (approved)': 5,
        'Curricular units 2nd sem (approved)': 4,
    }
    result = explain_student(semester_student, stage='semester')
    assert result['stage'] == 'semester'


# ---------- Artifact existence ----------

def test_shap_values_exist():
    assert os.path.exists('models/shap/admission_shap_values.npz')
    assert os.path.exists('models/shap/semester_shap_values.npz')


def test_fairness_report_exists():
    assert os.path.exists('reports/fairness_report.csv')


def test_mitigation_report_exists():
    assert os.path.exists('reports/mitigation_report.csv')


def test_mitigation_reduces_disparity():
    """Verify our mitigation actually reduced disparity."""
    df = pd.read_csv('reports/mitigation_report.csv')
    for _, row in df.iterrows():
        assert row['after_disparity'] < row['before_disparity'], \
            f"Mitigation failed for {row['stage']}/{row['attribute']}"