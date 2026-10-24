# End-to-End Modern Data Science Project Blueprint
### A Production-Grade Template for Pixi, Marimo, DuckDB, Scikit-Learn, and Hugging Face

This document is a reusable master playbook. You can use it to build and deploy any data analyst or data science portfolio project (healthcare, finance, customer churn, fraud detection, predictive maintenance) in under 30 minutes using this exact stack.

---

## 1. Architecture and Technology Stack

| Layer | Technology | Purpose | Key Benefit |
|---|---|---|---|
| **Environment** | **Pixi** (`pixi.toml`) | Conda-forge package manager and task runner | Reproducible cross-platform lockfile, zero system bloat |
| **LSP / Type Checking** | **Pyrefly** / `pyproject.toml` | Language server and IDE paths | Eliminates IDE import warnings by pointing to `.pixi/envs/default` |
| **Data Engine** | **DuckDB** | Columnar SQL database | Queries Parquet files directly, sub-50ms analytical aggregations |
| **Reactive Web App** | **Marimo** | Directed Acyclic Graph (DAG) notebook | Eliminates Streamlit script reruns, updates only modified cells |
| **Machine Learning** | **Scikit-Learn** | Pipeline, preprocessing, and model training | Leakage-free `ColumnTransformer` with calibrated thresholding |
| **Interactive Charts** | **Plotly** | Vectorized charting library | Interactive tooltips, zoom, and cross-cohort comparisons |
| **Verification** | **Pytest** | Automated unit test suite | Guarantees data bounds, SQL correctness, and ML risk contrasts |
| **Free Cloud Hosting** | **Hugging Face Spaces** | Public web application | Direct headless Marimo runner on port 7860 via Gradio container |

---

## 2. Directory Layout Blueprint

Create this file hierarchy for your project:

```text
my_data_project/
├── pixi.toml                # Environment dependencies and CLI tasks
├── pixi.lock                # Locked dependency tree
├── pyproject.toml           # IDE and LSP interpreter resolution
├── pyrefly.toml             # Pyrefly path configuration
├── README.md                # Humanized project presentation and metrics
├── LICENSE                  # MIT License
├── app.py                   # Reactive Marimo dashboard and live triage simulator
├── main.py                  # Production entrypoint for Hugging Face Spaces
├── requirements.txt         # Minimal dependency manifest for cloud hosting
├── marimo_widgets_showcase.py # Reference notebook documenting UI widgets
├── data/
│   └── dataset.parquet      # ZSTD compressed columnar dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # DuckDB ingestion, schema cleaning, and Parquet caching
│   ├── analysis.py          # KPI metrics, cohort aggregations, and custom SQL engine
│   ├── modeling.py          # Scikit-Learn pipelines, metric scoring, and inference
│   └── visualizations.py    # Plotly figure renderers with tailored palettes
└── tests/
    ├── __init__.py
    ├── test_data_loader.py  # Tests for data ingestion and mapping logic
    ├── test_analysis.py     # Tests for mathematical bounds and SQL aggregations
    ├── test_modeling.py     # Tests for pipeline evaluation and risk predictions
    └── test_visualizations.py # Tests for Plotly figure output structures
```

---

## 3. Configuration Files

### `pixi.toml`
Save in root:
```toml
[workspace]
name = "my-data-project"
version = "0.1.0"
description = "Production data science project with Marimo, DuckDB, and Scikit-Learn"
authors = ["Your Name <you@example.com>"]
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64", "win-64"]

[dependencies]
python = ">=3.11,<3.13"
duckdb = ">=1.0.0"
pandas = ">=2.0.0"
numpy = ">=1.24.0"
scikit-learn = ">=1.3.0"
plotly = ">=5.18.0"
marimo = ">=0.9.0"
pyarrow = ">=14.0.0"
pytest = ">=7.4.0"
requests = ">=2.31.0"

[tasks]
app = "marimo run app.py --port 8000"
edit = "marimo edit app.py"
widgets = "marimo run marimo_widgets_showcase.py --port 8001"
edit-widgets = "marimo edit marimo_widgets_showcase.py"
test = "pytest tests/ -v"
ingest = "python -m src.data_loader"
```

### `pyproject.toml` (Eliminates LSP / Pyrefly Import Errors)
Save in root:
```toml
[project]
name = "my-data-project"
version = "0.1.0"
dependencies = [
    "duckdb",
    "pandas",
    "numpy",
    "scikit-learn",
    "plotly",
    "marimo",
    "pyarrow",
    "pytest",
]

[tool.pyright]
venvPath = "."
venv = ".pixi/envs/default"
extraPaths = ["src"]
typeCheckingMode = "basic"
```

### `pyrefly.toml`
Save in root:
```toml
[environment]
python_path = ".pixi/envs/default/bin/python"
site_packages = [
    ".pixi/envs/default/lib/python3.12/site-packages",
    "src"
]
```

### `requirements.txt` (For Cloud Deployment)
Save in root:
```text
marimo>=0.11.0
duckdb>=1.1.0
pandas>=2.2.0
pyarrow>=17.0.0
scikit-learn>=1.5.0
plotly>=5.24.0
```

---

## 4. Production Cloud Runner: `main.py`

This script allows Marimo to run inside Hugging Face's free Gradio container without importing Gradio, websockets, or third-party wrappers:

```python
import os
import sys
import subprocess

# ZeroGPU validator check (if GPU hardware is ever selected)
try:
    import spaces
    @spaces.GPU
    def _zero_gpu_init():
        return True
    _zero_gpu_init()
except Exception:
    pass

if __name__ == "__main__":
    # Hugging Face exposes port 7860 to the public web
    port = os.environ.get("PORT", "7860")
    
    cmd = [
        sys.executable,
        "-m",
        "marimo",
        "run",
        "app.py",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--no-token",
        "--headless",
    ]
    print(f"Starting Marimo on 0.0.0.0:{port}...")
    sys.exit(subprocess.call(cmd))
```

---

## 5. Source Modules Blueprint (`src/`)

### A. Data Ingestion: `src/data_loader.py`
Use DuckDB to read Parquet directly, either from Hugging Face or a local cache:

```python
from pathlib import Path
from typing import Optional
import duckdb
import pandas as pd

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset.parquet"

def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
    except Exception:
        pass
    return con

def load_data(limit: Optional[int] = None) -> pd.DataFrame:
    con = get_duckdb_connection()
    if CACHE_PATH.exists():
        query = f"SELECT * FROM '{CACHE_PATH}'"
        if limit:
            query += f" LIMIT {limit}"
        return con.execute(query).df()
    
    # Fallback to local or remote extraction, then save to cache
    df = ... # Load or generate data
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, compression="zstd", index=False)
    return df
```

### B. Business Analytics and SQL: `src/analysis.py`
Build reusable cohort aggregations and an in-memory SQL query function:

```python
import pandas as pd
from src.data_loader import get_duckdb_connection

def compute_kpis(df: pd.DataFrame) -> dict:
    return {
        "total_records": len(df),
        "target_rate_pct": round(100.0 * df["target"].mean(), 1),
        "avg_duration": round(df["duration"].mean(), 2),
    }

def execute_custom_sql(df: pd.DataFrame, sql_query: str) -> pd.DataFrame:
    con = get_duckdb_connection()
    con.register("records", df)
    return con.execute(sql_query).df()
```

### C. Machine Learning Engine: `src/modeling.py`
Build a pipeline with class balancing and a real-time risk triage function:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

def train_models(df: pd.DataFrame) -> dict:
    X = df.drop(columns=["target"])
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["num_col"]),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), ["cat_col"]),
        ]
    )
    
    model = Pipeline([
        ("prep", preprocessor),
        ("clf", HistGradientBoostingClassifier(class_weight="balanced", random_state=42))
    ])
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "pipeline": model,
        "roc_auc": round(roc_auc_score(y_test, probs), 3),
        "pr_auc": round(average_precision_score(y_test, probs), 3),
    }

def predict_single_record(bundle: dict, inputs: dict) -> dict:
    df_single = pd.DataFrame([inputs])
    prob = float(bundle["pipeline"].predict_proba(df_single)[0, 1])
    tier = "High Risk" if prob >= 0.50 else "Moderate Risk" if prob >= 0.30 else "Low Risk"
    return {"probability": round(prob, 4), "tier": tier}
```

### D. Interactive Visualizations: `src/visualizations.py`
Return Plotly figure objects styled with a consistent design system (avoid browser defaults):

```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

CLINICAL_THEME = {
    "paper_bgcolor": "#ffffff",
    "plot_bgcolor": "#f8fafc",
    "font": {"family": "Inter, sans-serif", "color": "#0f172a"},
}

def plot_cohort_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(df, x="category", y="count", color_discrete_sequence=["#0d9488"])
    fig.update_layout(**CLINICAL_THEME, margin=dict(l=20, r=20, t=30, b=20))
    return fig
```

---

## 6. Marimo Reactive Dashboard Pattern (`app.py`)

Follow these core Marimo reactive rules:
1. **Rule 1**: Never access `widget.value` in the cell that created `widget = mo.ui...()`. Instantiate the widget in Cell A, and consume its value in downstream Cell B.
2. **Rule 2**: Use Marimo layout primitives (`mo.hstack`, `mo.vstack`, `mo.ui.tabs`) instead of string interpolation like `mo.md(f"{widget}")`.

```python
import marimo

app = marimo.App(width="full")

@app.cell
def imports():
    import marimo as mo
    from src.data_loader import load_data
    from src.analysis import compute_kpis, execute_custom_sql
    from src.modeling import train_models, predict_single_record
    from src.visualizations import plot_cohort_distribution
    return mo, load_data, compute_kpis, execute_custom_sql, train_models, predict_single_record, plot_cohort_distribution

@app.cell
def load_dataset(load_data):
    df_raw = load_data()
    return (df_raw,)

@app.cell
def filter_widgets(mo, df_raw):
    # Instantiate UI controls here
    category_dropdown = mo.ui.dropdown(
        options=["All"] + sorted(df_raw["category"].unique().tolist()),
        value="All",
        label="Filter Category:"
    )
    return (category_dropdown,)

@app.cell
def filter_data(df_raw, category_dropdown):
    # Consume UI controls in downstream cell
    if category_dropdown.value == "All":
        filtered_df = df_raw
    else:
        filtered_df = df_raw[df_raw["category"] == category_dropdown.value]
    return (filtered_df,)

@app.cell
def render_kpis(mo, compute_kpis, filtered_df):
    kpis = compute_kpis(filtered_df)
    return mo.hstack([
        mo.stat(value=f"{kpis['total_records']:,}", label="Cohort Records"),
        mo.stat(value=f"{kpis['target_rate_pct']}%", label="Event Rate"),
    ])
```

---

## 7. Test-Driven Verification Suite (`tests/`)

Write tests that prove data integrity and logical correctness:

```bash
pixi run test
```

### Essential Test Checks:
1. **Data conservation**: Row counts across SQL group-by operations must sum back to 100% of the sample.
2. **Mathematical bounds**: Rates and probabilities must fall strictly between `0` and `100` (or `0.0` and `1.0`).
3. **Clinical/business sanity**: An extreme high-risk scenario must output a higher risk probability than a baseline low-risk scenario (`assert high_risk["prob"] > low_risk["prob"]`).
4. **Figure rendering**: Plotly functions must return `isinstance(fig, go.Figure)` without throwing column errors.

---

## 8. Deployment to Hugging Face Spaces (5 Minutes)

### Step 1: Create the Space
1. Go to **huggingface.co/new-space**.
2. Select SDK: **Gradio** (Blank template).
3. Select Hardware: **CPU Basic (Free)**.
4. Visibility: **Public**.

### Step 2: Configure the Metadata
In your project `README.md`, include this header only for Hugging Face:
```yaml
---
title: My Project Name
emoji: 🚀
colorFrom: green
colorTo: blue
sdk: gradio
app_file: main.py
pinned: false
---
```

### Step 3: Login and Upload via CLI
Run in your terminal:
```bash
pixi run hf auth login
pixi run hf upload <your-username>/<your-space-name> . . --repo-type=space --exclude=".pixi/*" --exclude=".git/*"
```

The app will install packages and launch automatically at:  
`https://huggingface.co/spaces/<your-username>/<your-space-name>`

---

## 9. How to Adapt for a New Project (e.g., Credit Card Fraud, Churn)

To clone this architecture for a completely different dataset:

1. **Replace Data**: In `src/data_loader.py`, point to your new dataset (e.g., a customer churn CSV or fraud Parquet).
2. **Adjust Preprocessing**: Update `NUMERICAL_FEATURES` and `CATEGORICAL_FEATURES` in `src/modeling.py`.
3. **Change KPI Metrics**: Update `compute_kpis` in `src/analysis.py` to match the new business problem (e.g., Churn Rate %, Gross Volume $, Delinquency %).
4. **Adapt Simulator Sliders**: In `app.py`, change the real-time simulator inputs to match new features (e.g., Monthly Charges, Contract Length, Credit Score).
5. **Run Tests**: Execute `pixi run test` to verify math bounds.
6. **Upload**: Run `pixi run hf upload <user>/<new-space> . . --repo-type=space`. Done!
