"""Machine Learning Pipeline and Clinical Risk Triage Engine using Scikit-Learn."""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Feature Definitions
NUMERICAL_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_emergency",
    "number_inpatient",
]

CATEGORICAL_FEATURES = [
    "age_group",
    "gender",
    "admission_type",
    "primary_diagnosis_group",
    "glucose_test_result",
    "a1c_test_result",
    "change_in_meds",
]


def build_preprocessor() -> ColumnTransformer:
    """Build unified ColumnTransformer with scaling for numeric and one-hot for categorical."""
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def train_and_evaluate_models(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train baseline Logistic Regression and Gradient Boosting models, returning metrics and artifacts."""
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df["readmitted_30d"].copy()

    # Stratified split to preserve readmission class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()

    # Model 1: Logistic Regression (Interpretable Odds Ratios)
    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    # Model 2: HistGradientBoosting (Non-linear Feature Interactions)
    gb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    max_iter=150,
                    random_state=random_state,
                ),
            ),
        ]
    )

    models = {
        "Logistic Regression": lr_pipeline,
        "Gradient Boosting": gb_pipeline,
    }

    metrics = {}
    curves = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        # Clinical decision threshold: 0.35 (tuned for higher sensitivity in healthcare)
        threshold = 0.35
        y_pred = (y_prob >= threshold).astype(int)

        roc_auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_vals, precision_vals)

        metrics[name] = {
            "roc_auc": round(float(roc_auc), 3),
            "pr_auc": round(float(pr_auc), 3),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 3),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 3),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "test_size": len(y_test),
            "y_test": y_test,
            "y_prob": y_prob,
        }

        curves[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
            "recall": recall_vals,
            "precision": precision_vals,
            "pr_auc": pr_auc,
        }

    # Extract feature importances from Logistic Regression
    lr_model = models["Logistic Regression"].named_steps["classifier"]
    preproc = models["Logistic Regression"].named_steps["preprocessor"]
    cat_feature_names = preproc.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = NUMERICAL_FEATURES + list(cat_feature_names)
    coefficients = lr_model.coef_[0]

    feature_imp_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": np.abs(coefficients),
        "coefficient": coefficients,
    }).sort_values(by="importance", ascending=False)

    return {
        "models": models,
        "metrics": metrics,
        "curves": curves,
        "feature_importance": feature_imp_df,
        "best_model_name": "Gradient Boosting",
    }


def predict_patient_risk(
    bundle: Dict[str, Any],
    patient_data: Dict[str, Any],
    model_name: str = "Gradient Boosting",
    threshold: float = 0.35,
) -> Dict[str, Any]:
    """Calculate clinical readmission probability and actionable transition care recommendations."""
    pipeline = bundle["models"].get(model_name, list(bundle["models"].values())[0])

    patient_df = pd.DataFrame([patient_data])
    prob = float(pipeline.predict_proba(patient_df)[:, 1][0])

    # Classify clinical risk tier
    if prob < 0.20:
        tier = "Low Risk"
        recommendation = "Standard discharge protocols. Routine primary care follow-up within 14 days."
    elif prob < 0.40:
        tier = "Moderate Risk"
        recommendation = "Schedule clinical telephone check-in within 72 hours. Review patient medication adherence."
    else:
        tier = "High Risk"
        recommendation = "Urgent: Assign nurse case manager, schedule in-person clinic visit within 48h, and conduct comprehensive pharmacist medication reconciliation."

    # Identify individual risk drivers for this patient
    risk_drivers = []
    if patient_data.get("number_emergency", 0) > 0:
        risk_drivers.append(f"{patient_data['number_emergency']} Prior Emergency Visits")
    if patient_data.get("number_inpatient", 0) > 0:
        risk_drivers.append(f"{patient_data['number_inpatient']} Prior Inpatient Admissions")
    if patient_data.get("time_in_hospital", 0) >= 7:
        risk_drivers.append(f"Extended Length of Stay ({patient_data['time_in_hospital']} days)")
    if patient_data.get("change_in_meds") == "Yes":
        risk_drivers.append("Recent In-Hospital Medication Dosage Adjustment")
    if patient_data.get("a1c_test_result") in [">8", ">7"]:
        risk_drivers.append(f"Elevated HbA1c Glycemic Marker ({patient_data['a1c_test_result']})")
    if patient_data.get("primary_diagnosis_group") in ["Circulatory", "Diabetes", "Respiratory"]:
        risk_drivers.append(f"High-Risk Comorbidity: {patient_data['primary_diagnosis_group']}")

    if not risk_drivers:
        risk_drivers.append("Low comorbidity profile with standard inpatient duration")

    return {
        "risk_probability": round(prob, 3),
        "risk_percentage": round(prob * 100, 1),
        "risk_tier": tier,
        "predicted_readmitted": 1 if prob >= threshold else 0,
        "clinical_recommendation": recommendation,
        "risk_drivers": risk_drivers,
    }
