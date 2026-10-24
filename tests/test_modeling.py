"""Tests for machine learning pipeline and clinical risk prediction module."""
import pytest
import pandas as pd
import numpy as np
from src.modeling import (
    train_and_evaluate_models,
    predict_patient_risk,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.data_loader import generate_synthetic_clinical_cohort

@pytest.fixture
def clinical_df():
    return generate_synthetic_clinical_cohort(n_records=400, seed=99)

def test_train_and_evaluate_models(clinical_df):
    bundle = train_and_evaluate_models(clinical_df)
    assert "models" in bundle
    assert "Logistic Regression" in bundle["models"]
    assert "Gradient Boosting" in bundle["models"]
    assert "metrics" in bundle
    assert "feature_importance" in bundle

    metrics_lr = bundle["metrics"]["Logistic Regression"]
    assert 0.5 <= metrics_lr["roc_auc"] <= 1.0
    assert 0.0 <= metrics_lr["f1"] <= 1.0
    assert metrics_lr["confusion_matrix"].shape == (2, 2)

    imp_df = bundle["feature_importance"]
    assert isinstance(imp_df, pd.DataFrame)
    assert len(imp_df) > 0

def test_predict_patient_risk(clinical_df):
    bundle = train_and_evaluate_models(clinical_df)

    # Test typical low-risk patient
    low_risk_patient = {
        "age_group": "[40-50)",
        "gender": "Female",
        "admission_type": "Elective",
        "time_in_hospital": 2,
        "num_lab_procedures": 25,
        "num_procedures": 1,
        "num_medications": 8,
        "number_emergency": 0,
        "number_inpatient": 0,
        "primary_diagnosis_group": "Musculoskeletal",
        "glucose_test_result": "Norm",
        "a1c_test_result": "Norm",
        "change_in_meds": "No",
    }
    low_res = predict_patient_risk(bundle, low_risk_patient)
    assert "risk_probability" in low_res
    assert 0.0 <= low_res["risk_probability"] <= 1.0
    assert low_res["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert "clinical_recommendation" in low_res

    # Test high-risk patient
    high_risk_patient = {
        "age_group": "[70-80)",
        "gender": "Male",
        "admission_type": "Emergency",
        "time_in_hospital": 10,
        "num_lab_procedures": 80,
        "num_procedures": 3,
        "num_medications": 28,
        "number_emergency": 3,
        "number_inpatient": 2,
        "primary_diagnosis_group": "Circulatory",
        "glucose_test_result": ">300",
        "a1c_test_result": ">8",
        "change_in_meds": "Yes",
    }
    high_res = predict_patient_risk(bundle, high_risk_patient)
    assert high_res["risk_probability"] > low_res["risk_probability"]
    assert high_res["risk_tier"] in ["Moderate Risk", "High Risk"]
