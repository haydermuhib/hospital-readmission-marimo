import marimo

__generated_with = "0.9.0"
app = marimo.App(
    width="full",
    app_title="Marimo UI Components & Reactive Widgets Showcase",
    css_file=None,
)


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go

    return mo, pd, np, go


@app.cell
def _(mo):
    # Header & Architectural Overview
    header = mo.md(
        """
        <div style="background: #ffffff; border-bottom: 1px solid #e2e8f0; padding: 24px 32px; margin: -16px -16px 24px -16px; border-radius: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                <div>
                    <span style="background: #0d9488; color: #ffffff; font-size: 11px; font-weight: 700; letter-spacing: 0.05em; padding: 3px 8px; border-radius: 4px; text-transform: uppercase;">Reference Guide & Gallery</span>
                    <h1 style="font-size: 26px; font-weight: 800; color: #0f172a; margin: 8px 0 4px 0;">Marimo Reactive UI Widgets & Components Catalog</h1>
                    <p style="font-size: 14px; color: #64748b; margin: 0; max-width: 820px;">
                        Interactive guide documenting every Marimo UI widget utilized in the <b>CardioPulse Hospital Readmission & Clinical Risk Intelligence</b> application. Explore live controls, reactive state inspectors, and production best practices.
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 12px; color: #64748b;">Marimo Reactivity Rule</div>
                    <div style="font-size: 12px; font-weight: 600; color: #0f172a; background: #f8fafc; padding: 6px 12px; border-radius: 6px; border: 1px solid #e2e8f0;">Cell A: Create widget → Cell B: Access .value</div>
                </div>
            </div>
        </div>
        """
    )
    return (header,)


# ---------------------------------------------------------------------------
# WIDGET 1: mo.ui.dropdown
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    # Create dropdown widget (Cell 1)
    demo_dropdown = mo.ui.dropdown(
        options=["Circulatory (Cardiac/Vascular)", "Respiratory (COPD/Pneumonia)", "Diabetes & Endocrine", "Genitourinary", "Digestive"],
        value="Circulatory (Cardiac/Vascular)",
        label="Select Diagnostic Category:",
    )
    return (demo_dropdown,)


@app.cell
def _(mo, demo_dropdown):
    # Consume dropdown value (Cell 2)
    dropdown_doc = mo.vstack(
        [
            mo.md(
                """
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h3 style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 0;">1. <code>mo.ui.dropdown</code> (Single-Select Menu)</h3>
                        <span style="font-size: 11px; background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 4px; font-weight: 600;">Selection Control</span>
                    </div>
                    <p style="font-size: 13px; color: #475569; margin: 0 0 14px 0;">
                        Provides a dropdown selection menu from an iterable of options (or a dictionary of label-to-value pairs).
                    </p>
                    <div style="background: #f8fafc; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0; margin-bottom: 12px;">
                        <b>Project Usage:</b> Selected primary diagnoses (e.g., <i>Circulatory</i>, <i>Diabetes</i>), admission urgency, patient gender, and HbA1c test markers.
                    </div>
                </div>
                """
            ),
            mo.hstack(
                [
                    demo_dropdown,
                    mo.md(f"<div style='background: #f1f5f9; padding: 10px 14px; border-radius: 6px; font-size: 13px;'><b>Live Selected Value:</b> <code style='color: #0d9488; font-weight: 700;'>{demo_dropdown.value}</code></div>"),
                ],
                align="center",
                gap=1.5,
            ),
        ],
        gap=0.8,
    )
    return (dropdown_doc,)


# ---------------------------------------------------------------------------
# WIDGET 2: mo.ui.slider
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    # Create slider widget (Cell 1)
    demo_slider = mo.ui.slider(
        start=1,
        stop=14,
        step=1,
        value=6,
        label="Length of Hospital Stay (Days):",
    )
    return (demo_slider,)


@app.cell
def _(mo, demo_slider):
    # Consume slider value (Cell 2)
    slider_doc = mo.vstack(
        [
            mo.md(
                """
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h3 style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 0;">2. <code>mo.ui.slider</code> (Numeric Range Slider)</h3>
                        <span style="font-size: 11px; background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 4px; font-weight: 600;">Numeric Control</span>
                    </div>
                    <p style="font-size: 13px; color: #475569; margin: 0 0 14px 0;">
                        Renders an interactive range slider for bounded integer or floating-point values with step size control.
                    </p>
                    <div style="background: #f8fafc; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0; margin-bottom: 12px;">
                        <b>Project Usage:</b> Filtering max length of stay (1–14 days), simulating prior emergency room encounters (0–6), and setting inpatient medication counts (1–45).
                    </div>
                </div>
                """
            ),
            mo.hstack(
                [
                    demo_slider,
                    mo.md(f"<div style='background: #f1f5f9; padding: 10px 14px; border-radius: 6px; font-size: 13px;'><b>Live Slider Value:</b> <code style='color: #0d9488; font-weight: 700;'>{demo_slider.value} Days</code></div>"),
                ],
                align="center",
                gap=1.5,
            ),
        ],
        gap=0.8,
    )
    return (slider_doc,)


# ---------------------------------------------------------------------------
# WIDGET 3: mo.ui.text_area
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    # Create text_area widget (Cell 1)
    demo_sql_input = mo.ui.text_area(
        value="SELECT primary_diagnosis_group, count(*) AS count FROM patients GROUP BY 1",
        label="Dynamic SQL Query Editor:",
        rows=3,
    )
    return (demo_sql_input,)


@app.cell
def _(mo, demo_sql_input):
    # Consume text_area value (Cell 2)
    textarea_doc = mo.vstack(
        [
            mo.md(
                """
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h3 style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 0;">3. <code>mo.ui.text_area</code> (Multiline Code / Text Editor)</h3>
                        <span style="font-size: 11px; background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 4px; font-weight: 600;">Text / Code Input</span>
                    </div>
                    <p style="font-size: 13px; color: #475569; margin: 0 0 14px 0;">
                        A resizable multiline text input field that re-evaluates downstream reactive cells whenever the user updates the text.
                    </p>
                    <div style="background: #f8fafc; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0; margin-bottom: 12px;">
                        <b>Project Usage:</b> Powered the interactive DuckDB SQL Sandbox tab, allowing clinicians and analysts to run ad-hoc SQL queries on the active patient cohort.
                    </div>
                </div>
                """
            ),
            demo_sql_input,
            mo.md(f"<div style='background: #f1f5f9; padding: 10px 14px; border-radius: 6px; font-size: 12px;'><b>Active Text Content:</b> <pre style='margin: 4px 0 0 0; color: #0369a1;'>{demo_sql_input.value}</pre></div>"),
        ],
        gap=0.8,
    )
    return (textarea_doc,)


# ---------------------------------------------------------------------------
# WIDGET 4: mo.ui.table
# ---------------------------------------------------------------------------
@app.cell
def _(mo, pd):
    # Interactive Table Widget Demo
    sample_clinical_table = pd.DataFrame({
        "Patient ID": [10001, 10002, 10003, 10004, 10005],
        "Diagnosis": ["Circulatory", "Diabetes", "Respiratory", "Genitourinary", "Circulatory"],
        "Stay (Days)": [4, 7, 2, 5, 8],
        "Prior Emergency Visits": [1, 2, 0, 0, 3],
        "30-Day Readmission Risk": ["22.4%", "38.1%", "12.8%", "18.0%", "45.2%"],
    })

    demo_table = mo.ui.table(sample_clinical_table, page_size=5)

    table_doc = mo.vstack(
        [
            mo.md(
                """
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h3 style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 0;">4. <code>mo.ui.table</code> (Interactive Data Grid)</h3>
                        <span style="font-size: 11px; background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 4px; font-weight: 600;">Data Presentation</span>
                    </div>
                    <p style="font-size: 13px; color: #475569; margin: 0 0 14px 0;">
                        Renders an interactive DataFrame with client-side column sorting, pagination, and row selection capabilities.
                    </p>
                    <div style="background: #f8fafc; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0; margin-bottom: 12px;">
                        <b>Project Usage:</b> Displaying dynamic DuckDB SQL query execution outputs and tabular patient stay escalation breakdowns.
                    </div>
                </div>
                """
            ),
            demo_table,
        ],
        gap=0.8,
    )
    return sample_clinical_table, demo_table, table_doc


# ---------------------------------------------------------------------------
# WIDGET 5: mo.ui.plotly
# ---------------------------------------------------------------------------
@app.cell
def _(mo, go):
    # Plotly Interactive Figure Demo
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Circulatory", "Diabetes", "Respiratory", "Digestive", "Genitourinary"],
            y=[21.9, 22.2, 20.5, 17.7, 15.1],
            marker=dict(color=["#e11d48", "#e11d48", "#e11d48", "#0d9488", "#0d9488"]),
            text=["21.9%", "22.2%", "20.5%", "17.7%", "15.1%"],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Readmission Rate: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="30-Day Hospital Readmission Rate by Primary Comorbidity (%)",
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=50, b=40),
        font=dict(family="Inter, sans-serif", size=12, color="#475569"),
    )
    fig.update_yaxes(range=[0, 28], gridcolor="#f1f5f9")
    fig.update_xaxes(gridcolor="#f1f5f9")

    demo_plotly = mo.ui.plotly(fig)

    plotly_doc = mo.vstack(
        [
            mo.md(
                """
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h3 style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 0;">5. <code>mo.ui.plotly</code> (Interactive Chart Wrapper)</h3>
                        <span style="font-size: 11px; background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 4px; font-weight: 600;">Data Visualization</span>
                    </div>
                    <p style="font-size: 13px; color: #475569; margin: 0 0 14px 0;">
                        Seamlessly integrates Plotly WebGL and SVG figures into the Marimo reactivity graph. Supports responsive resizing and click/hover events.
                    </p>
                    <div style="background: #f8fafc; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0; margin-bottom: 12px;">
                        <b>Project Usage:</b> Visualizing comorbidity risk bars, clinical utilization scatter matrices, ROC/PR curves, and medical confusion matrices.
                    </div>
                </div>
                """
            ),
            demo_plotly,
        ],
        gap=0.8,
    )
    return fig, demo_plotly, plotly_doc


# ---------------------------------------------------------------------------
# WIDGET 6: mo.ui.tabs & Layout Containers
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    demo_tabs = mo.ui.tabs(
        {
            "Executive Overview": mo.md("<div style='padding: 16px; background: #ffffff; border-radius: 6px;'><b>Executive View:</b> Tracks hospital-wide 30-day readmission benchmarks and bed utilization.</div>"),
            "Clinical Triage": mo.md("<div style='padding: 16px; background: #ffffff; border-radius: 6px;'><b>Clinical Triage View:</b> Classifies high-risk patient cohorts for targeted post-discharge phone calls and home nurse visits.</div>"),
            "API Specs": mo.md("<div style='padding: 16px; background: #ffffff; border-radius: 6px;'><b>Technical View:</b> Powered by DuckDB vectorized columnar queries over Apache Parquet.</div>"),
        }
    )

    tabs_doc = mo.vstack(
        [
            mo.md(
                """
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h3 style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 0;">6. <code>mo.ui.tabs</code> & Native Layout (<code>mo.hstack</code> / <code>mo.vstack</code>)</h3>
                        <span style="font-size: 11px; background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 4px; font-weight: 600;">Layout & Structure</span>
                    </div>
                    <p style="font-size: 13px; color: #475569; margin: 0 0 14px 0;">
                        Native flexbox and tabbed containers that partition complex analytics dashboards into clean, responsive sections without string interpolation bugs.
                    </p>
                    <div style="background: #f8fafc; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0; margin-bottom: 12px;">
                        <b>Project Usage:</b> Divided the main dashboard into Diagnostic Cohorts, Glycemic Testing, and DuckDB SQL Sandbox views.
                    </div>
                </div>
                """
            ),
            demo_tabs,
        ],
        gap=0.8,
    )
    return demo_tabs, tabs_doc


# ---------------------------------------------------------------------------
# MAIN ASSEMBLY
# ---------------------------------------------------------------------------
@app.cell
def _(
    mo,
    header,
    dropdown_doc,
    slider_doc,
    textarea_doc,
    table_doc,
    plotly_doc,
    tabs_doc,
):
    catalog_view = mo.vstack(
        [
            header,
            dropdown_doc,
            slider_doc,
            textarea_doc,
            table_doc,
            plotly_doc,
            tabs_doc,
        ],
        gap=2.0,
    )
    catalog_view
    return (catalog_view,)


if __name__ == "__main__":
    app.run()
