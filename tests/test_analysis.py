"""Tests for clinical analysis and cohort analytics module."""
import pytest
import pandas as pd
from src.analysis import (
    compute_kpis,
    readmission_by_cohort,
    medication_change_impact,
    length_of_stay_risk_profile,
    execute_custom_sql
)
from src.data_loader import generate_synthetic_clinical_cohort

@pytest.fixture
def sample_df():
    return generate_synthetic_clinical_cohort(n_records=300, seed=123)

def test_compute_kpis(sample_df):
    kpis = compute_kpis(sample_df)
    assert "total_patients" in kpis
    assert kpis["total_patients"] == 300
    assert 0 <= kpis["readmission_rate_pct"] <= 100
    assert kpis["median_stay_days"] > 0
    assert kpis["avg_medications"] > 0
    assert kpis["high_risk_patients"] >= 0

def test_readmission_by_cohort(sample_df):
    cohort_df = readmission_by_cohort(sample_df, "primary_diagnosis_group")
    assert isinstance(cohort_df, pd.DataFrame)
    assert "primary_diagnosis_group" in cohort_df.columns
    assert "patient_count" in cohort_df.columns
    assert "readmission_rate_pct" in cohort_df.columns
    assert len(cohort_df) > 0
    assert (cohort_df["readmission_rate_pct"] >= 0).all()

def test_medication_change_impact(sample_df):
    med_df = medication_change_impact(sample_df)
    assert isinstance(med_df, pd.DataFrame)
    assert "change_in_meds" in med_df.columns
    assert "readmission_rate_pct" in med_df.columns

def test_length_of_stay_risk_profile(sample_df):
    stay_df = length_of_stay_risk_profile(sample_df)
    assert isinstance(stay_df, pd.DataFrame)
    assert "stay_bracket" in stay_df.columns
    assert "readmission_rate_pct" in stay_df.columns

def test_execute_custom_sql(sample_df):
    query = """
    SELECT primary_diagnosis_group, count(*) AS count
    FROM patients
    GROUP BY primary_diagnosis_group
    ORDER BY count DESC
    """
    sql_res = execute_custom_sql(sample_df, query)
    assert isinstance(sql_res, pd.DataFrame)
    assert len(sql_res) > 0
    assert sql_res["count"].sum() == 300
