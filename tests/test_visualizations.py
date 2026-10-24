"""Tests for Plotly visualization generation module."""
import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.visualizations import (
    plot_readmission_by_category,
    plot_risk_matrix,
    plot_medication_impact,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    plot_feature_importance,
)
from src.data_loader import generate_synthetic_clinical_cohort
from src.analysis import readmission_by_cohort, medication_change_impact

@pytest.fixture
def sample_df():
    return generate_synthetic_clinical_cohort(n_records=250, seed=42)

def test_plot_readmission_by_category(sample_df):
    cohort_df = readmission_by_cohort(sample_df, "primary_diagnosis_group")
    fig = plot_readmission_by_category(cohort_df, "primary_diagnosis_group", "Readmission by Diagnosis")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0

def test_plot_risk_matrix(sample_df):
    fig = plot_risk_matrix(sample_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0

def test_plot_medication_impact(sample_df):
    med_df = medication_change_impact(sample_df)
    fig = plot_medication_impact(med_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0

def test_plot_confusion_matrix():
    cm = np.array([[180, 20], [15, 35]])
    fig = plot_confusion_matrix(cm, ["Not Readmitted", "Readmitted <30d"])
    assert isinstance(fig, go.Figure)

def test_plot_roc_pr_curves():
    fpr = np.linspace(0, 1, 50)
    tpr = np.sqrt(fpr)
    recall = np.linspace(0, 1, 50)
    precision = 1 - 0.5 * recall
    fig = plot_roc_pr_curves(fpr, tpr, 0.82, recall, precision, 0.65)
    assert isinstance(fig, go.Figure)

def test_plot_feature_importance():
    imp_df = pd.DataFrame({
        "feature": ["number_inpatient", "number_emergency", "time_in_hospital", "a1c_high"],
        "importance": [0.35, 0.28, 0.18, 0.12]
    })
    fig = plot_feature_importance(imp_df)
    assert isinstance(fig, go.Figure)
