"""Clinical Analytics and Cohort Computation Module using DuckDB and Pandas."""
from typing import Dict, Any
import duckdb
import numpy as np
import pandas as pd


def execute_custom_sql(df: pd.DataFrame, sql_query: str) -> pd.DataFrame:
    """Execute an arbitrary SQL query on the hospital DataFrame using DuckDB."""
    con = duckdb.connect(database=":memory:")
    con.register("patients", df)
    return con.execute(sql_query).df()


def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute executive hospital inpatient KPIs and clinical outcome metrics."""
    if df.empty:
        return {
            "total_patients": 0,
            "readmission_rate_pct": 0.0,
            "readmitted_count": 0,
            "median_stay_days": 0.0,
            "avg_medications": 0.0,
            "high_risk_patients": 0,
            "emergency_encounter_rate_pct": 0.0,
            "avg_lab_procedures": 0.0,
        }

    total_patients = len(df)
    readmitted_count = int(df["readmitted_30d"].sum())
    readmission_rate_pct = round((readmitted_count / total_patients) * 100, 2)
    median_stay_days = round(float(df["time_in_hospital"].median()), 1)
    avg_medications = round(float(df["num_medications"].mean()), 1)
    avg_lab_procedures = round(float(df["num_lab_procedures"].mean()), 1)

    # High risk: patients with >=1 prior emergency visits OR >=1 prior inpatient stays
    prior_utilizers = df[(df["number_emergency"] > 0) | (df["number_inpatient"] > 0)]
    high_risk_patients = int(len(prior_utilizers))

    emergency_encounter_rate_pct = round(
        (len(df[df["number_emergency"] > 0]) / total_patients) * 100, 2
    )

    return {
        "total_patients": total_patients,
        "readmission_rate_pct": readmission_rate_pct,
        "readmitted_count": readmitted_count,
        "median_stay_days": median_stay_days,
        "avg_medications": avg_medications,
        "avg_lab_procedures": avg_lab_procedures,
        "high_risk_patients": high_risk_patients,
        "emergency_encounter_rate_pct": emergency_encounter_rate_pct,
    }


def readmission_by_cohort(df: pd.DataFrame, cohort_col: str) -> pd.DataFrame:
    """Calculate 30-day readmission rate grouped by clinical cohort column."""
    if df.empty or cohort_col not in df.columns:
        return pd.DataFrame(columns=[cohort_col, "patient_count", "readmitted_count", "readmission_rate_pct"])

    query = f"""
    SELECT
        {cohort_col},
        count(*) AS patient_count,
        sum(readmitted_30d) AS readmitted_count,
        round(100.0 * sum(readmitted_30d) / count(*), 2) AS readmission_rate_pct
    FROM patients
    GROUP BY {cohort_col}
    ORDER BY patient_count DESC
    """
    con = duckdb.connect(database=":memory:")
    con.register("patients", df)
    result = con.execute(query).df()
    return result


def medication_change_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze the impact of medication changes and A1C testing on readmissions."""
    if df.empty:
        return pd.DataFrame()

    query = """
    SELECT
        change_in_meds,
        a1c_test_result,
        count(*) AS patient_count,
        sum(readmitted_30d) AS readmitted_count,
        round(100.0 * sum(readmitted_30d) / count(*), 2) AS readmission_rate_pct
    FROM patients
    GROUP BY change_in_meds, a1c_test_result
    ORDER BY change_in_meds, patient_count DESC
    """
    con = duckdb.connect(database=":memory:")
    con.register("patients", df)
    return con.execute(query).df()


def length_of_stay_risk_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Segment patients by length of stay brackets and compute risk escalation."""
    if df.empty:
        return pd.DataFrame()

    query = """
    SELECT
        CASE
            WHEN time_in_hospital BETWEEN 1 AND 2 THEN '1-2 Days (Short)'
            WHEN time_in_hospital BETWEEN 3 AND 4 THEN '3-4 Days (Moderate)'
            WHEN time_in_hospital BETWEEN 5 AND 7 THEN '5-7 Days (Extended)'
            ELSE '8+ Days (Prolonged)'
        END AS stay_bracket,
        CASE
            WHEN time_in_hospital BETWEEN 1 AND 2 THEN 1
            WHEN time_in_hospital BETWEEN 3 AND 4 THEN 2
            WHEN time_in_hospital BETWEEN 5 AND 7 THEN 3
            ELSE 4
        END AS sort_order,
        count(*) AS patient_count,
        round(avg(num_medications), 1) AS avg_medications,
        round(avg(num_lab_procedures), 1) AS avg_lab_procedures,
        sum(readmitted_30d) AS readmitted_count,
        round(100.0 * sum(readmitted_30d) / count(*), 2) AS readmission_rate_pct
    FROM patients
    GROUP BY stay_bracket, sort_order
    ORDER BY sort_order ASC
    """
    con = duckdb.connect(database=":memory:")
    con.register("patients", df)
    return con.execute(query).df()
