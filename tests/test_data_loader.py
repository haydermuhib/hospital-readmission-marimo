"""Tests for data ingestion and cleaning module."""
import pytest
import pandas as pd
import numpy as np
import duckdb
from src.data_loader import (
    get_duckdb_connection,
    categorize_icd9,
    clean_hospital_data,
    load_hospital_data,
    generate_synthetic_clinical_cohort
)

def test_duckdb_connection():
    """Verify DuckDB creates an in-memory connection with httpfs loaded."""
    con = get_duckdb_connection()
    assert con is not None
    # Verify DuckDB executes SQL queries
    res = con.execute("SELECT 42 AS value").fetchone()
    assert res[0] == 42

def test_categorize_icd9():
    """Test ICD-9 diagnosis code grouping into clinical categories."""
    assert categorize_icd9("410") == "Circulatory"     # Myocardial infarction
    assert categorize_icd9("428.0") == "Circulatory"   # Heart failure
    assert categorize_icd9("493") == "Respiratory"     # Asthma
    assert categorize_icd9("250.01") == "Diabetes"     # Diabetes
    assert categorize_icd9("584") == "Genitourinary"   # Renal failure
    assert categorize_icd9("820") == "Injury/Trauma"   # Femur fracture
    assert categorize_icd9("?") == "Other/Unknown"
    assert categorize_icd9(None) == "Other/Unknown"

def test_generate_synthetic_clinical_cohort():
    """Test that synthetic cohort generator creates realistic hospital rows."""
    df = generate_synthetic_clinical_cohort(n_records=100)
    assert len(df) == 100
    expected_cols = [
        "patient_id", "age_group", "gender", "admission_type",
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_emergency", "number_inpatient",
        "primary_diagnosis_group", "glucose_test_result", "a1c_test_result",
        "change_in_meds", "readmitted_30d"
    ]
    for col in expected_cols:
        assert col in df.columns

    # Verify target variable binary values
    assert set(df["readmitted_30d"].unique()).issubset({0, 1})
    # Verify stay length is positive
    assert (df["time_in_hospital"] > 0).all()

def test_clean_hospital_data():
    """Test dataframe standardization and sanitization."""
    raw = pd.DataFrame({
        "encounter_id": [101, 102],
        "patient_nbr": [1001, 1002],
        "age": ["[70-80)", "[50-60)"],
        "gender": ["Female", "Male"],
        "admission_type_id": [1, 2],
        "time_in_hospital": [4, 7],
        "num_lab_procedures": [45, 62],
        "num_procedures": [1, 3],
        "num_medications": [15, 22],
        "number_emergency": [0, 2],
        "number_inpatient": [1, 0],
        "diag_1": ["428", "250"],
        "max_glu_serum": ["None", ">300"],
        "A1Cresult": [">8", "None"],
        "change": ["Ch", "No"],
        "readmitted": ["<30", "NO"]
    })
    cleaned = clean_hospital_data(raw)
    assert "readmitted_30d" in cleaned.columns
    assert cleaned.loc[0, "readmitted_30d"] == 1
    assert cleaned.loc[1, "readmitted_30d"] == 0
    assert cleaned.loc[0, "primary_diagnosis_group"] == "Circulatory"
    assert cleaned.loc[1, "primary_diagnosis_group"] == "Diabetes"
