# Week 3 Summary — Model Training, Tuning & Selection

**Author:** Prateek Veeresh Aggimath
**Dates:** 14-10-2026 to 15-10-2026

## Accomplishments
- ✅ Trained 3 baseline models (LogReg, RF, XGBoost) at 2 stages = 6 models
- ✅ 5-fold cross-validation comparison of all 6 models
- ✅ Hyperparameter tuning via RandomizedSearchCV
- ✅ **Fixed critical SMOTE leakage bug** — retrained with SMOTE inside CV
- ✅ Selected production models for admission and semester stages
- ✅ Generated final metrics, confusion matrices, ROC curves

## Final Production Models
- **Admission stage:** RandomForest (tuned) — `production_admission.pkl`
- **Semester stage:** RandomForest (tuned) — `production_semester.pkl`

## Final Test Metrics (honest, no leakage)

| Stage | Accuracy | Macro F1 | ROC-AUC | Dropout Recall |
|-------|----------|----------|---------|----------------|
| Admission | 0.6068 | 0.5474 | 0.7547 | 0.6056 |
| Semester | 0.7616 | 0.7162 | 0.8836 | 0.7007 |

## Model Comparison (5-fold CV, honest)

| Model | Admission F1 | Semester F1 |
|-------|--------------|-------------|
| LogReg | 0.5277 | 0.7040 |
| RandomForest (baseline) | 0.5480 | 0.7059 |
| XGBoost (baseline) | 0.5532 | 0.7010 |
| **RF (tuned)** | **0.5619** | 0.7151 |
| **XGB (tuned)** | **0.5622** | **0.7193** |

## Key Insight — SMOTE Leakage Fix

**Problem discovered:** Initial tuning applied SMOTE to the full training set BEFORE CV splits. This caused synthetic samples to leak across folds, inflating CV F1 to 0.74 (admission) and 0.83 (semester).

**Solution:** Moved SMOTE **inside** the CV loop using `imblearn.pipeline.Pipeline`. Each fold's training portion is SMOTE-augmented, validation fold stays raw.

**Result:** CV scores dropped to honest values:
- Admission F1: 0.74 → 0.56
- Semester F1: 0.83 → 0.72

**Lesson:** Always apply resampling **inside** CV, never before splitting. This is a common beginner mistake and worth documenting.

## Two-Stage Design Validation
- **Admission stage:** catches 60% of dropouts early — more time for intervention
- **Semester stage:** catches 70% of dropouts — more accurate but later
- Both useful in different intervention windows

## Files Created
| File | Purpose |
|------|---------|
| `src/models/train_logreg.py` | Baseline LogReg training |
| `src/models/train_rf.py` | Baseline RandomForest |
| `src/models/train_xgb.py` | Baseline XGBoost |
| `src/models/evaluate.py` | 5-fold CV comparison |
| `src/models/tune.py` | Hyperparameter tuning (SMOTE inside CV) |
| `src/models/select_best.py` | Final selection + eval artifacts |

## Report Artifacts
- `reports/model_comparison.csv`
- `reports/model_comparison.png`
- `reports/tuning_results.json`
- `reports/final_metrics.csv`
- `reports/figures/confusion_matrix_admission.png`
- `reports/figures/confusion_matrix_semester.png`
- `reports/figures/roc_curves_admission.png`
- `reports/figures/roc_curves_semester.png`

## Blockers / Notes
- Initial SMOTE leakage bug — fixed and documented
- RF tuning gains were modest (+0.01–0.02 F1)
- XGBoost slightly better in CV but RF chosen for smaller model size

## Time Spent
~18 hours across 6 days

## Plan for Week 4
- SHAP global feature importance
- SHAP per-student explanations
- Build `explain_student()` function for Flask
- Fairness audit (gender, age, scholarship)
- Bias mitigation strategies