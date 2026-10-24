"""Interactive Plotly Data Visualizations for Hospital Analytics & Clinical Decision Support."""
from typing import List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Medical / Clinical Design Palette
COLOR_TEAL = "#0d9488"        # Primary Clinical Accent
COLOR_CRIMSON = "#e11d48"     # Alert / High-Risk / Readmitted
COLOR_SLATE_DARK = "#1e293b"  # Deep Text & Primary Series
COLOR_SLATE_MUTED = "#64748b" # Secondary Axis & Labels
COLOR_GRID = "#f1f5f9"        # Faint hairline grid
COLOR_CARD_BG = "#ffffff"     # White surface


def _apply_clinical_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    """Apply consistent, quiet, modern clinical typography and spacing."""
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 15, "color": COLOR_SLATE_DARK, "family": "Inter, system-ui, sans-serif"},
        },
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=30, t=50, b=40),
        font={"family": "Inter, system-ui, sans-serif", "color": COLOR_SLATE_MUTED, "size": 12},
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#e2e8f0",
            font_size=12,
            font_color=COLOR_SLATE_DARK,
            font_family="Inter, system-ui, sans-serif",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color=COLOR_SLATE_DARK),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=COLOR_GRID,
        linecolor="#cbd5e1",
        tickfont=dict(size=11, color=COLOR_SLATE_MUTED),
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=COLOR_GRID,
        linecolor="#cbd5e1",
        tickfont=dict(size=11, color=COLOR_SLATE_MUTED),
    )
    return fig


def plot_readmission_by_category(
    cohort_df: pd.DataFrame, category_col: str, title: str
) -> go.Figure:
    """Horizontal bar chart showing readmission rate by patient cohort with count badges."""
    df_sorted = cohort_df.sort_values(by="readmission_rate_pct", ascending=True).copy()

    # Color highest risk bars in alert crimson
    colors = [
        COLOR_CRIMSON if r >= 20.0 else COLOR_TEAL
        for r in df_sorted["readmission_rate_pct"]
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df_sorted[category_col].astype(str),
            x=df_sorted["readmission_rate_pct"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{r:.1f}% ({c:,} pts)" for r, c in zip(df_sorted["readmission_rate_pct"], df_sorted["patient_count"])],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Readmission Rate: %{x:.2f}%<br><extra></extra>",
        )
    )

    _apply_clinical_layout(fig, title, height=max(320, len(df_sorted) * 38))
    fig.update_xaxes(title="30-Day Readmission Rate (%)", range=[0, max(df_sorted["readmission_rate_pct"].max() * 1.35, 15)])
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_risk_matrix(df: pd.DataFrame) -> go.Figure:
    """Clinical utilization landscape: Length of stay vs Lab procedures colored by readmission."""
    sample_df = df.sample(min(len(df), 1200), random_state=42) if len(df) > 1200 else df

    fig = go.Figure()

    # Non-readmitted cohort
    not_readmitted = sample_df[sample_df["readmitted_30d"] == 0]
    fig.add_trace(
        go.Scatter(
            x=not_readmitted["time_in_hospital"],
            y=not_readmitted["num_lab_procedures"],
            mode="markers",
            name="Discharged Safely",
            marker=dict(
                size=6,
                color="#94a3b8",
                opacity=0.45,
                line=dict(width=0),
            ),
            hovertemplate="Stay: %{x} days<br>Lab Procedures: %{y}<br>Outcome: No Readmission<extra></extra>",
        )
    )

    # Readmitted cohort (<30 days)
    readmitted = sample_df[sample_df["readmitted_30d"] == 1]
    fig.add_trace(
        go.Scatter(
            x=readmitted["time_in_hospital"],
            y=readmitted["num_lab_procedures"],
            mode="markers",
            name="Readmitted < 30 Days",
            marker=dict(
                size=8,
                color=COLOR_CRIMSON,
                opacity=0.85,
                line=dict(width=1, color="#ffffff"),
            ),
            hovertemplate="Stay: %{x} days<br>Lab Procedures: %{y}<br>Outcome: Readmitted <30d<extra></extra>",
        )
    )

    _apply_clinical_layout(fig, "Clinical Utilization: Stay Duration vs. Lab Testing Volume", height=390)
    fig.update_xaxes(title="Length of Inpatient Stay (Days)", dtick=2)
    fig.update_yaxes(title="Number of Lab Procedures")
    return fig


def plot_medication_impact(med_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart showing readmission rate by A1C test result and medication change."""
    fig = go.Figure()

    for change_val, color in [("Yes", COLOR_CRIMSON), ("No", COLOR_TEAL)]:
        sub = med_df[med_df["change_in_meds"] == change_val].sort_values(by="a1c_test_result")
        fig.add_trace(
            go.Bar(
                name=f"Medication Adjusted: {change_val}",
                x=sub["a1c_test_result"],
                y=sub["readmission_rate_pct"],
                marker_color=color,
                text=[f"{val:.1f}%" for val in sub["readmission_rate_pct"]],
                textposition="outside",
                hovertemplate="<b>A1C Result: %{x}</b><br>Med Change: " + change_val + "<br>Readmission Rate: %{y:.2f}%<extra></extra>",
            )
        )

    _apply_clinical_layout(fig, "Readmission Risk by Glycemic Testing (A1C) & Medication Adjustments", height=360)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Glycated Hemoglobin (HbA1c) Result")
    fig.update_yaxes(title="30-Day Readmission Rate (%)", range=[0, max(med_df["readmission_rate_pct"].max() * 1.25, 20)])
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> go.Figure:
    """Medical confusion matrix heatmap with counts and true positive rates."""
    total = np.sum(cm)
    percentages = (cm / total) * 100

    annotations = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            pct = percentages[i, j]
            text_color = "#ffffff" if (i == 1 and j == 1) or (i == 0 and j == 0) else COLOR_SLATE_DARK
            annotations.append(
                dict(
                    x=class_names[j],
                    y=class_names[i],
                    text=f"<b>{count:,}</b><br>({pct:.1f}%)",
                    showarrow=False,
                    font=dict(color=text_color, size=13),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale=[[0.0, "#f8fafc"], [0.5, "#cbd5e1"], [1.0, COLOR_TEAL]],
            showscale=False,
        )
    )

    _apply_clinical_layout(fig, "Model Decision Matrix (Confusion Heatmap)", height=340)
    fig.update_layout(annotations=annotations)
    fig.update_xaxes(title="Clinically Predicted Outcome", side="bottom")
    fig.update_yaxes(title="Actual Patient Outcome", autorange="reversed")
    return fig


def plot_roc_pr_curves(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    recall: np.ndarray,
    precision: np.ndarray,
    pr_auc: float,
) -> go.Figure:
    """Side-by-side ROC and Precision-Recall clinical calibration curves."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"ROC Curve (AUC = {roc_auc:.3f})",
            f"Precision-Recall Curve (PR-AUC = {pr_auc:.3f})",
        ),
    )

    # ROC trace
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name="ROC Curve",
            line=dict(color=COLOR_TEAL, width=2.5),
            hovertemplate="FPR: %{x:.3f}<br>TPR (Recall): %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    # Random diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance Baseline",
            line=dict(color="#cbd5e1", width=1.5, dash="dash"),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Precision-Recall trace
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name="Precision-Recall",
            line=dict(color=COLOR_CRIMSON, width=2.5),
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    _apply_clinical_layout(fig, "Model Discriminative Performance & Triage Tradeoffs", height=360)
    fig.update_xaxes(title="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title="True Positive Rate (Sensitivity)", row=1, col=1)
    fig.update_xaxes(title="Recall (Sensitivity)", row=1, col=2)
    fig.update_yaxes(title="Precision (PPV)", row=1, col=2)
    fig.update_layout(showlegend=False)
    return fig


def plot_feature_importance(imp_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart showing the highest clinical risk predictor variables."""
    top_df = imp_df.sort_values(by="importance", ascending=True).tail(top_n).copy()

    # Clean feature names for clinician readability
    name_cleaner = {
        "number_inpatient": "Prior Inpatient Admissions",
        "number_emergency": "Prior Emergency Room Visits",
        "time_in_hospital": "Length of Hospital Stay",
        "num_medications": "Number of Medications",
        "num_lab_procedures": "Volume of Lab Procedures",
        "num_procedures": "Specialist Procedures",
        "change_in_meds_Yes": "Medication Dosage Adjusted",
        "primary_diagnosis_group_Circulatory": "Diagnosis: Heart/Circulatory",
        "primary_diagnosis_group_Diabetes": "Diagnosis: Diabetes",
        "primary_diagnosis_group_Respiratory": "Diagnosis: Respiratory",
        "a1c_test_result_>8": "A1C Elevated (>8.0%)",
        "age_group_[70-80)": "Age Bracket: 70-80",
    }
    labels = [name_cleaner.get(f, f.replace("_", " ").title()) for f in top_df["feature"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=top_df["importance"],
            orientation="h",
            marker=dict(
                color=COLOR_TEAL,
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>Predictive Weight: %{x:.4f}<extra></extra>",
        )
    )

    _apply_clinical_layout(fig, f"Top {top_n} Predictive Clinical Risk Factors", height=max(320, top_n * 30))
    fig.update_xaxes(title="Relative Feature Importance (Gini / Logistic Coeff)")
    return fig
