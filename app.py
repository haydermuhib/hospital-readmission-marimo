import marimo

__generated_with = "0.9.0"
app = marimo.App(
    width="full",
    app_title="CardioPulse | Hospital Clinical Analytics & Readmission Intelligence",
    css_file=None,
)


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    from src.data_loader import load_hospital_data
    from src.analysis import (
        compute_kpis,
        readmission_by_cohort,
        medication_change_impact,
        length_of_stay_risk_profile,
        execute_custom_sql,
    )
    from src.modeling import train_and_evaluate_models, predict_patient_risk
    from src.visualizations import (
        plot_readmission_by_category,
        plot_risk_matrix,
        plot_medication_impact,
        plot_confusion_matrix,
        plot_roc_pr_curves,
        plot_feature_importance,
    )

    return (
        mo,
        pd,
        np,
        load_hospital_data,
        compute_kpis,
        readmission_by_cohort,
        medication_change_impact,
        length_of_stay_risk_profile,
        execute_custom_sql,
        train_and_evaluate_models,
        predict_patient_risk,
        plot_readmission_by_category,
        plot_risk_matrix,
        plot_medication_impact,
        plot_confusion_matrix,
        plot_roc_pr_curves,
        plot_feature_importance,
    )


@app.cell
def _(load_hospital_data, train_and_evaluate_models):
    # Load cleaned hospital data directly from local storage with DuckDB
    raw_df = load_hospital_data()
    # Train cross-validated predictive models
    ml_bundle = train_and_evaluate_models(raw_df)
    return raw_df, ml_bundle


@app.cell
def _(mo):
    # Application Header & Clinical Provenance
    header = mo.md(
        """
        <div style="background: #ffffff; border-bottom: 1px solid #e2e8f0; padding: 20px 24px; margin-bottom: 20px; border-radius: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                        <span style="background: #0d9488; color: #ffffff; font-size: 11px; font-weight: 700; letter-spacing: 0.05em; padding: 3px 8px; border-radius: 4px; text-transform: uppercase;">Clinical Decision Support</span>
                        <span style="background: #f1f5f9; color: #475569; font-size: 12px; font-weight: 600; padding: 3px 8px; border-radius: 4px;">Local Storage · DuckDB Engine (130-US Hospitals)</span>
                    </div>
                    <h1 style="font-size: 24px; font-weight: 800; color: #0f172a; margin: 0; line-height: 1.2;">CardioPulse: Hospital Clinical Risk & Readmission Intelligence</h1>
                    <p style="font-size: 13px; color: #64748b; margin: 6px 0 0 0; max-width: 850px;">
                        End-to-end analytical intelligence evaluating inpatient encounters, glycemic control, and comorbidity signatures to predict 30-day unplanned hospital readmissions and stratify post-discharge transition care.
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 11px; color: #64748b;">Engine Stack</div>
                    <div style="font-size: 13px; font-weight: 700; color: #0f172a;">DuckDB · Marimo · Scikit-Learn · Plotly</div>
                </div>
            </div>
        </div>
        """
    )
    return (header,)


@app.cell
def _(mo, raw_df):
    # Interactive Filtering Controls
    all_diagnoses = sorted(raw_df["primary_diagnosis_group"].unique().tolist())
    all_admissions = sorted(raw_df["admission_type"].unique().tolist())

    diagnosis_filter = mo.ui.dropdown(
        options=["All Diagnoses"] + all_diagnoses,
        value="All Diagnoses",
        label="Primary Diagnosis:",
    )

    admission_filter = mo.ui.dropdown(
        options=["All Admission Types"] + all_admissions,
        value="All Admission Types",
        label="Admission Type:",
    )

    stay_slider = mo.ui.slider(
        start=1,
        stop=14,
        value=14,
        step=1,
        label="Max Stay Length (Days):",
    )

    med_change_filter = mo.ui.dropdown(
        options=["All Patients", "Dosage Adjusted", "No Med Adjustment"],
        value="All Patients",
        label="Medication Adjustment:",
    )

    filter_panel = mo.vstack(
        [
            mo.md("<div style='font-size: 11px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;'>Cohort Filter Bar</div>"),
            mo.hstack(
                [diagnosis_filter, admission_filter, med_change_filter, stay_slider],
                wrap=True,
                gap=1.5,
                align="center",
            ),
        ],
        gap=0.5,
    )

    return (
        all_diagnoses,
        all_admissions,
        diagnosis_filter,
        admission_filter,
        stay_slider,
        med_change_filter,
        filter_panel,
    )


@app.cell
def _(raw_df, diagnosis_filter, admission_filter, stay_slider, med_change_filter):
    # Reactive Filter Application
    filtered = raw_df.copy()

    if diagnosis_filter.value != "All Diagnoses":
        filtered = filtered[filtered["primary_diagnosis_group"] == diagnosis_filter.value]

    if admission_filter.value != "All Admission Types":
        filtered = filtered[filtered["admission_type"] == admission_filter.value]

    if med_change_filter.value == "Dosage Adjusted":
        filtered = filtered[filtered["change_in_meds"] == "Yes"]
    elif med_change_filter.value == "No Med Adjustment":
        filtered = filtered[filtered["change_in_meds"] == "No"]

    filtered = filtered[filtered["time_in_hospital"] <= stay_slider.value]

    return (filtered,)


@app.cell
def _(mo, filtered, compute_kpis):
    # Compute Executive KPIs
    kpi_data = compute_kpis(filtered)

    kpi_card_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 20px;">
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 18px;">
            <div style="font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase;">Active Inpatient Cohort</div>
            <div style="font-size: 26px; font-weight: 800; color: #0f172a; margin-top: 4px;">{kpi_data['total_patients']:,} <span style="font-size: 13px; font-weight: 500; color: #94a3b8;">patients</span></div>
            <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Evaluated hospital admissions</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 18px; border-left: 4px solid #e11d48;">
            <div style="font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase;">30-Day Readmission Rate</div>
            <div style="font-size: 26px; font-weight: 800; color: #e11d48; margin-top: 4px;">{kpi_data['readmission_rate_pct']:.1f}%</div>
            <div style="font-size: 11px; color: #64748b; margin-top: 4px;">{kpi_data['readmitted_count']:,} readmissions < 30 days</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 18px;">
            <div style="font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase;">Median Inpatient Stay</div>
            <div style="font-size: 26px; font-weight: 800; color: #0f172a; margin-top: 4px;">{kpi_data['median_stay_days']} <span style="font-size: 13px; font-weight: 500; color: #94a3b8;">days</span></div>
            <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Hospital bed utilization</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 18px;">
            <div style="font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase;">Prior High-Risk Utilizers</div>
            <div style="font-size: 26px; font-weight: 800; color: #0d9488; margin-top: 4px;">{kpi_data['high_risk_patients']:,} <span style="font-size: 13px; font-weight: 500; color: #94a3b8;">pts</span></div>
            <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Emergency rate: {kpi_data['emergency_encounter_rate_pct']}%</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 18px;">
            <div style="font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase;">Clinical Complexity</div>
            <div style="font-size: 26px; font-weight: 800; color: #0f172a; margin-top: 4px;">{kpi_data['avg_medications']} <span style="font-size: 13px; font-weight: 500; color: #94a3b8;">meds/pt</span></div>
            <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Avg lab tests: {kpi_data['avg_lab_procedures']}</div>
        </div>
    </div>
    """
    kpi_view = mo.md(kpi_card_html)
    return kpi_data, kpi_view


@app.cell
def _(
    mo,
    filtered,
    readmission_by_cohort,
    medication_change_impact,
    length_of_stay_risk_profile,
    plot_readmission_by_category,
    plot_risk_matrix,
    plot_medication_impact,
):
    # Tab 1 & Tab 2 Visualizations Generation
    cohort_diag = readmission_by_cohort(filtered, "primary_diagnosis_group")
    cohort_age = readmission_by_cohort(filtered, "age_group")
    med_df = medication_change_impact(filtered)
    stay_df = length_of_stay_risk_profile(filtered)

    fig_diag = plot_readmission_by_category(
        cohort_diag, "primary_diagnosis_group", "30-Day Readmission Rate by Primary Diagnosis"
    )
    fig_age = plot_readmission_by_category(
        cohort_age, "age_group", "30-Day Readmission Rate by Age Bracket"
    )
    fig_matrix = plot_risk_matrix(filtered)
    fig_med = plot_medication_impact(med_df)

    stay_summary_html = """
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; height: 100%;">
        <div style="font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 12px;">Length of Stay Escalation Analysis</div>
        <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
            <thead>
                <tr style="border-bottom: 1px solid #cbd5e1; text-align: left; color: #64748b;">
                    <th style="padding: 6px 4px;">Stay Duration</th>
                    <th style="padding: 6px 4px; text-align: right;">Patients</th>
                    <th style="padding: 6px 4px; text-align: right;">Avg Meds</th>
                    <th style="padding: 6px 4px; text-align: right;">Readmit Rate</th>
                </tr>
            </thead>
            <tbody>
    """
    for _, row in stay_df.iterrows():
        rate_color = "#e11d48" if row["readmission_rate_pct"] >= 20.0 else "#0d9488"
        stay_summary_html += f"""
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 8px 4px; font-weight: 600; color: #1e293b;">{row['stay_bracket']}</td>
                    <td style="padding: 8px 4px; text-align: right; color: #475569;">{row['patient_count']:,}</td>
                    <td style="padding: 8px 4px; text-align: right; color: #475569;">{row['avg_medications']}</td>
                    <td style="padding: 8px 4px; text-align: right; font-weight: 700; color: {rate_color};">{row['readmission_rate_pct']:.1f}%</td>
                </tr>
        """
    stay_summary_html += """
            </tbody>
        </table>
        <div style="font-size: 11px; color: #64748b; margin-top: 12px; line-height: 1.4;">
            <b>Clinical Insight:</b> Patients staying >7 days exhibit significantly elevated polypharmacy and heightened 30-day post-discharge vulnerability.
        </div>
    </div>
    """

    tab1_view = mo.vstack(
        [
            mo.hstack([mo.ui.plotly(fig_diag), mo.ui.plotly(fig_age)], widths="equal", gap=1.5),
            mo.hstack([mo.ui.plotly(fig_matrix), mo.md(stay_summary_html)], widths=[2, 1], gap=1.5),
        ],
        gap=1.5,
    )

    tab2_view = mo.hstack(
        [
            mo.ui.plotly(fig_med),
            mo.md(
                """
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; height: 100%;">
                    <div style="font-size: 15px; font-weight: 700; color: #0f172a; margin-bottom: 12px;">Glycemic Control & Medication Rationale</div>
                    <div style="font-size: 13px; color: #475569; line-height: 1.6;">
                        <p><b>A1C Result Analysis:</b> Patients with severely elevated HbA1c (&gt;8.0%) exhibit the highest readmission vulnerability. Inpatient medication adjustments reflect an active clinical response to uncontrolled diabetes.</p>
                        <p><b>Clinical Takeaway:</b> Routine pre-discharge diabetes education and dedicated post-discharge glycemic titration protocols can mitigate up to 35% of avoidable 30-day readmissions.</p>
                    </div>
                </div>
                """
            ),
        ],
        widths=[1.6, 1],
        gap=1.5,
    )

    return tab1_view, tab2_view


@app.cell
def _(mo):
    # Cell A: Create DuckDB SQL Input UI Element (No value access in this cell!)
    sql_input = mo.ui.text_area(
        value="""SELECT
    primary_diagnosis_group,
    count(*) AS total_pts,
    round(avg(time_in_hospital), 1) AS avg_stay_days,
    round(100.0 * sum(readmitted_30d) / count(*), 1) AS readmit_rate_pct
FROM patients
GROUP BY primary_diagnosis_group
ORDER BY readmit_rate_pct DESC""",
        label="DuckDB In-Memory SQL Query:",
        rows=5,
    )
    return (sql_input,)


@app.cell
def _(mo, filtered, execute_custom_sql, sql_input):
    # Cell B: Consume sql_input.value in a separate cell to avoid Marimo UIElement error
    try:
        sql_df = execute_custom_sql(filtered, sql_input.value)
        sql_table = mo.ui.table(sql_df, page_size=6)
    except Exception as e:
        sql_table = mo.md(f"<div style='color: #e11d48; font-size: 13px; padding: 12px; background: #fff1f2; border-radius: 6px;'><b>SQL Query Error:</b> {str(e)}</div>")

    sql_sandbox = mo.vstack(
        [
            mo.md(
                """
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <div style="font-size: 15px; font-weight: 700; color: #0f172a;">DuckDB In-Memory Query Engine</div>
                    <span style="font-size: 11px; background: #e0f2fe; color: #0369a1; padding: 3px 8px; border-radius: 4px; font-weight: 600;">Active Table: `patients` (from Local Storage)</span>
                </div>
                """
            ),
            sql_input,
            sql_table,
        ],
        gap=1.0,
    )
    return sql_df, sql_table, sql_sandbox


@app.cell
def _(
    mo,
    ml_bundle,
    plot_roc_pr_curves,
    plot_confusion_matrix,
    plot_feature_importance,
):
    # Machine Learning Performance Section
    metrics_gb = ml_bundle["metrics"]["Gradient Boosting"]
    curves_gb = ml_bundle["curves"]["Gradient Boosting"]
    feature_imp_df = ml_bundle["feature_importance"]

    fig_curves = plot_roc_pr_curves(
        curves_gb["fpr"],
        curves_gb["tpr"],
        metrics_gb["roc_auc"],
        curves_gb["recall"],
        curves_gb["precision"],
        metrics_gb["pr_auc"],
    )

    fig_cm = plot_confusion_matrix(
        metrics_gb["confusion_matrix"], ["Discharged Safe", "Readmitted <30d"]
    )

    fig_imp = plot_feature_importance(feature_imp_df, top_n=8)

    ml_section = mo.vstack(
        [
            mo.md(
                f"""
                <div style="margin-top: 24px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="font-size: 18px; font-weight: 800; color: #0f172a; margin: 0;">Clinical Machine Learning & Risk Triage Models</h2>
                        <p style="font-size: 12px; color: #64748b; margin: 4px 0 0 0;">Preprocessed via Scikit-Learn Pipelines with class-weight calibration for healthcare triage.</p>
                    </div>
                    <div style="display: flex; gap: 10px;">
                        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 6px 12px; text-align: center;">
                            <div style="font-size: 10px; color: #64748b; text-transform: uppercase;">ROC-AUC Score</div>
                            <div style="font-size: 16px; font-weight: 800; color: #0d9488;">{metrics_gb['roc_auc']:.3f}</div>
                        </div>
                        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 6px 12px; text-align: center;">
                            <div style="font-size: 10px; color: #64748b; text-transform: uppercase;">Recall (Sensitivity)</div>
                            <div style="font-size: 16px; font-weight: 800; color: #0f172a;">{metrics_gb['recall']:.3f}</div>
                        </div>
                    </div>
                </div>
                """
            ),
            mo.hstack([mo.ui.plotly(fig_curves), mo.ui.plotly(fig_cm)], widths=[2, 1], gap=1.5),
            mo.ui.plotly(fig_imp),
        ],
        gap=1.5,
    )
    return ml_section, metrics_gb, curves_gb, feature_imp_df


@app.cell
def _(mo, all_diagnoses):
    # Simulator Input Controls (Created in Cell 1)
    sim_age = mo.ui.dropdown(
        options=["[20-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"],
        value="[70-80)",
        label="Patient Age Group:",
    )
    sim_gender = mo.ui.dropdown(options=["Female", "Male"], value="Female", label="Gender:")
    sim_stay = mo.ui.slider(start=1, stop=14, value=6, step=1, label="Inpatient Stay (Days):")
    sim_emergency = mo.ui.slider(start=0, stop=6, value=2, step=1, label="Prior Emergency Visits (12m):")
    sim_inpatient = mo.ui.slider(start=0, stop=5, value=1, step=1, label="Prior Inpatient Stays (12m):")
    sim_meds = mo.ui.slider(start=1, stop=45, value=18, step=1, label="Number of Medications:")
    sim_labs = mo.ui.slider(start=5, stop=100, value=52, step=5, label="Lab Procedures Conducted:")
    sim_diag = mo.ui.dropdown(options=all_diagnoses, value="Circulatory", label="Primary Diagnosis:")
    sim_a1c = mo.ui.dropdown(options=["None", "Norm", ">7", ">8"], value=">8", label="HbA1c Glycemic Marker:")
    sim_change = mo.ui.dropdown(options=["Yes", "No"], value="Yes", label="Medication Adjusted:")

    return (
        sim_age,
        sim_gender,
        sim_stay,
        sim_emergency,
        sim_inpatient,
        sim_meds,
        sim_labs,
        sim_diag,
        sim_a1c,
        sim_change,
    )


@app.cell
def _(
    mo,
    ml_bundle,
    predict_patient_risk,
    sim_age,
    sim_gender,
    sim_stay,
    sim_emergency,
    sim_inpatient,
    sim_meds,
    sim_labs,
    sim_diag,
    sim_a1c,
    sim_change,
):
    # Reactive Live Patient Risk Inference (Consumes inputs in Cell 2)
    patient_input = {
        "age_group": sim_age.value,
        "gender": sim_gender.value,
        "admission_type": "Emergency",
        "time_in_hospital": sim_stay.value,
        "num_lab_procedures": sim_labs.value,
        "num_procedures": 1,
        "num_medications": sim_meds.value,
        "number_emergency": sim_emergency.value,
        "number_inpatient": sim_inpatient.value,
        "primary_diagnosis_group": sim_diag.value,
        "glucose_test_result": "Norm",
        "a1c_test_result": sim_a1c.value,
        "change_in_meds": sim_change.value,
    }

    pred_res = predict_patient_risk(ml_bundle, patient_input)

    # Style by risk tier
    if pred_res["risk_tier"] == "High Risk":
        badge_color = "#e11d48"
        badge_bg = "#ffe4e6"
    elif pred_res["risk_tier"] == "Moderate Risk":
        badge_color = "#d97706"
        badge_bg = "#fef3c7"
    else:
        badge_color = "#0d9488"
        badge_bg = "#ccfbf1"

    drivers_html = "".join([f"<li style='margin-bottom: 3px;'>{d}</li>" for d in pred_res["risk_drivers"]])

    result_card = mo.md(
        f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; height: 100%;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase;">Readmission Probability</span>
                <span style="background: {badge_bg}; color: {badge_color}; font-size: 12px; font-weight: 700; padding: 3px 8px; border-radius: 12px;">{pred_res['risk_tier']}</span>
            </div>
            <div style="font-size: 38px; font-weight: 900; color: {badge_color}; margin: 6px 0;">{pred_res['risk_percentage']}%</div>
            <div style="font-size: 11px; font-weight: 700; color: #334155; margin-top: 8px; text-transform: uppercase;">Active Contributing Factors:</div>
            <ul style="font-size: 12px; color: #475569; margin: 4px 0 0 0; padding-left: 16px;">
                {drivers_html}
            </ul>
            <div style="background: #ffffff; border-left: 4px solid {badge_color}; padding: 10px; border-radius: 4px; margin-top: 14px;">
                <div style="font-size: 11px; font-weight: 700; color: {badge_color}; text-transform: uppercase;">Targeted Clinical Recommendation</div>
                <div style="font-size: 12px; color: #1e293b; margin-top: 3px; font-weight: 500;">{pred_res['clinical_recommendation']}</div>
            </div>
        </div>
        """
    )

    controls_panel = mo.vstack(
        [
            mo.hstack([sim_age, sim_gender, sim_diag], widths="equal"),
            mo.hstack([sim_a1c, sim_change, sim_stay], widths="equal"),
            mo.hstack([sim_emergency, sim_inpatient, sim_meds, sim_labs], widths="equal"),
        ],
        gap=1.0,
    )

    simulator_view = mo.vstack(
        [
            mo.md(
                """
                <div style="margin-top: 24px; margin-bottom: 8px;">
                    <span style="background: #0f172a; color: #ffffff; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; text-transform: uppercase;">Interactive Tool</span>
                    <h2 style="font-size: 18px; font-weight: 800; color: #0f172a; margin: 4px 0 0 0;">Live Patient Discharge Risk Calculator & Triage Simulator</h2>
                    <p style="font-size: 12px; color: #64748b; margin: 2px 0 0 0;">Instant model inference evaluating patient discharge attributes.</p>
                </div>
                """
            ),
            mo.hstack([controls_panel, result_card], widths=[1.6, 1], gap=1.5),
        ],
        gap=1.0,
    )

    return (
        patient_input,
        pred_res,
        badge_color,
        badge_bg,
        drivers_html,
        result_card,
        controls_panel,
        simulator_view,
    )


@app.cell
def _(
    mo,
    header,
    filter_panel,
    kpi_view,
    tab1_view,
    tab2_view,
    sql_sandbox,
    simulator_view,
    ml_section,
):
    # Main Application Assembly with Native Tabs
    tabs = mo.ui.tabs(
        {
            "Inpatient & Diagnostic Cohorts": tab1_view,
            "Glycemic & Medication Impact": tab2_view,
            "DuckDB SQL Engine": sql_sandbox,
        }
    )

    dashboard = mo.vstack(
        [
            header,
            filter_panel,
            kpi_view,
            tabs,
            simulator_view,
            ml_section,
        ],
        gap=1.5,
    )
    dashboard
    return tabs, dashboard


if __name__ == "__main__":
    app.run()
