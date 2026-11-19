from __future__ import annotations

from io import BytesIO
import hashlib
import html
import secrets
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modules.auth import register_user
from modules.workspace import (
    create_project,
    delete_project,
    list_activities,
    list_projects,
    rename_project,
    set_project_favorite,
)


st.set_page_config(
    page_title="DatasetTest Overview | Turn Data Into Intelligence",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #f5f7ff;
        --muted: #9da9c5;
        --panel: rgba(20, 29, 52, .78);
        --line: rgba(164, 183, 255, .13);
        --lime: #c9f36a;
        --purple: #a78bfa;
        --cyan: #67e8f9;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(86, 62, 173, .23), transparent 34%),
            radial-gradient(ellipse at 94% 22%, rgba(37, 151, 167, .12), transparent 30%),
            #0a1020;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #121a31 0%, #0e1528 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: var(--ink); }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--ink) !important; letter-spacing: -.035em; }
    p, label, [data-testid="stMarkdownContainer"] { color: #d7def1; }
    [data-testid="stMetric"] {
        min-height: 135px;
        padding: 20px 22px;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: linear-gradient(145deg, rgba(28, 39, 68, .92), rgba(17, 26, 47, .88));
        box-shadow: 0 16px 40px rgba(0, 0, 0, .13);
    }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .88rem; }
    [data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 750; }
    [data-testid="stMetricDelta"] { color: var(--lime) !important; }
    [data-testid="stFileUploader"], [data-testid="stSelectbox"], [data-testid="stSlider"] {
        border-radius: 14px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 16px;
        overflow: hidden;
    }
    .hero {
        position: relative;
        overflow: hidden;
        margin: 0 0 1.65rem;
        padding: 28px 32px;
        border: 1px solid rgba(187, 199, 255, .17);
        border-radius: 24px;
        background: linear-gradient(115deg, rgba(34, 43, 76, .94), rgba(19, 38, 57, .88));
        box-shadow: 0 22px 65px rgba(0, 0, 0, .2);
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        right: 7%;
        top: -150px;
        border-radius: 50%;
        background: rgba(201, 243, 106, .22);
        filter: blur(55px);
    }
    .hero-kicker {
        color: var(--lime);
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero-title {
        margin: 8px 0 5px;
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 850;
        letter-spacing: -.055em;
        line-height: 1.05;
    }
    .hero-title span { color: var(--lime); }
    .hero-copy { margin: 0; color: #b6c0d8; font-size: 1rem; }
    .hero-developer { margin: .35rem 0 0; color: var(--muted); font-size: .75rem; }
    .section-label {
        margin: 1.8rem 0 .7rem;
        color: var(--muted);
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .stButton > button, .stDownloadButton > button {
        border: 0;
        border-radius: 12px;
        background: var(--lime);
        color: #172012;
        font-weight: 800;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border: 0;
        background: #d8ff83;
        color: #172012;
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(201, 243, 106, .18);
    }
    [data-testid="stRadio"] > label { color: var(--muted) !important; }
    [data-testid="stRadio"] [role="radiogroup"] label {
        padding: 10px 12px;
        margin: 3px 0;
        border: 1px solid transparent;
        border-radius: 12px;
        transition: background .15s ease, border-color .15s ease;
    }
    [data-testid="stRadio"] [role="radiogroup"] label:hover {
        background: rgba(167, 139, 250, .10);
        border-color: var(--line);
    }
    [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        background: rgba(201, 243, 106, .11);
        border-color: rgba(201, 243, 106, .28);
    }
    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.1rem;
        padding: 12px 16px;
        border: 1px solid var(--line);
        border-radius: 15px;
        background: rgba(20, 29, 52, .58);
    }
    .topbar-brand { color: var(--ink); font-weight: 800; }
    .topbar-meta { color: var(--muted); font-size: .83rem; }
    .quick-card {
        height: 100%;
        padding: 18px;
        border: 1px solid var(--line);
        border-radius: 17px;
        background: var(--panel);
    }
    .quick-card strong { color: var(--lime); }
    div[data-testid="stPlotlyChart"] {
        padding: 10px;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: var(--panel);
    }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def make_sample_data(rows: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    tenure = rng.integers(1, 72, rows)
    monthly_charges = np.clip(rng.normal(72, 22, rows), 20, 150).round(2)
    support_calls = rng.poisson(1.4, rows).clip(0, 9)
    payment_delay = rng.integers(0, 16, rows)
    usage_gb = np.clip(rng.normal(38, 17, rows), 2, 110).round(1)
    add_on_count = rng.integers(0, 5, rows)
    contract = rng.choice(["Bulanan", "Tahunan", "Dua tahun"], rows, p=[0.58, 0.27, 0.15])
    payment_method = rng.choice(["Kartu kredit", "Transfer bank", "E-wallet"], rows, p=[0.42, 0.33, 0.25])

    logit = (
        1.15
        - 0.035 * tenure
        + 0.028 * (monthly_charges - 70)
        + 0.48 * support_calls
        + 0.055 * payment_delay
        - 0.28 * add_on_count
        + np.where(contract == "Bulanan", 0.85, 0)
        + np.where(contract == "Dua tahun", -0.75, 0)
        + rng.normal(0, 0.75, rows)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "Lama berlangganan (bulan)": tenure,
            "Biaya bulanan": monthly_charges,
            "Panggilan ke support": support_calls,
            "Keterlambatan bayar (hari)": payment_delay,
            "Pemakaian data (GB)": usage_gb,
            "Jumlah add-on": add_on_count,
            "Jenis kontrak": contract,
            "Metode pembayaran": payment_method,
            "Churn": np.where(churn == 1, "Berhenti", "Aktif"),
        }
    )
    return df


@st.cache_data
def load_uploaded_data(contents: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".xlsx"):
        return pd.read_excel(BytesIO(contents), engine="openpyxl")
    return pd.read_csv(BytesIO(contents))


def validate_dataset(dataframe: pd.DataFrame) -> None:
    if dataframe.empty or len(dataframe.columns) < 2:
        raise ValueError("File harus berisi data dan setidaknya dua kolom.")
    if dataframe.columns.has_duplicates:
        raise ValueError("Nama kolom harus unik.")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_cols),
            ("categorical", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def metric_value(metric: str, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray | None, classes: np.ndarray) -> float:
    average = "binary" if len(classes) == 2 else "weighted"
    pos_label = classes[-1] if len(classes) == 2 else 1

    if metric == "Accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "Precision":
        return float(precision_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0))
    if metric == "Recall":
        return float(recall_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0))
    if metric == "F1-score":
        return float(f1_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0))
    if y_prob is None:
        return float("nan")
    if len(classes) == 2:
        return float(roc_auc_score(y_true == classes[-1], y_prob[:, 1]))
    return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))


def evaluate_models(
    data: pd.DataFrame,
    target: str,
    test_size: float,
    folds: int,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    df = data.dropna(subset=[target]).copy()
    X = df.drop(columns=[target])
    y = df[target].astype(str)
    classes = np.sort(y.unique())

    if len(classes) < 2:
        raise ValueError("Target harus memiliki minimal dua kelas.")
    if y.value_counts().min() < 2:
        raise ValueError("Setiap kelas target harus memiliki minimal dua baris data.")
    if X.shape[1] == 0:
        raise ValueError("CSV harus memiliki setidaknya satu kolom fitur.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    prediction_results: dict[str, np.ndarray] = {}

    splits = min(folds, int(y.value_counts().min()))
    if splits < 2:
        raise ValueError("Tidak cukup data per kelas untuk cross-validation.")

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X)),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
        if len(classes) == 2 and prob is not None:
            pred = np.where(prob[:, 1] >= threshold, classes[-1], classes[0])
        else:
            pred = pipeline.predict(X_test)

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)

        results.append(
            {
                "Model": model_name,
                "Accuracy": metric_value("Accuracy", y_test, pred, prob, classes),
                "Precision": metric_value("Precision", y_test, pred, prob, classes),
                "Recall": metric_value("Recall", y_test, pred, prob, classes),
                "F1-score": metric_value("F1-score", y_test, pred, prob, classes),
                "ROC-AUC": metric_value("ROC-AUC", y_test, pred, prob, classes),
                "CV Accuracy": float(cv_scores.mean()),
                "CV Std": float(cv_scores.std()),
            }
        )
        fitted_models[model_name] = pipeline
        prediction_results[model_name] = pred

    df_results = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    prediction_df = pd.DataFrame(
        {
            "Baris data": y_test.index.to_numpy() + 1,
            "Aktual": y_test.to_numpy(),
            **prediction_results,
        }
    ).reset_index(drop=True)
    return df_results, fitted_models, prediction_df


if "guest_workspace_owner" not in st.session_state:
    guest_username = f"guest_{secrets.token_hex(12)}"
    register_user(guest_username, secrets.token_urlsafe(32))
    st.session_state.guest_workspace_owner = guest_username
guest_user = str(st.session_state.guest_workspace_owner)

user_projects = list_projects(guest_user)
project_names = {
    str(project["project_id"]): str(project["name"]) for project in user_projects
}
active_project_id = st.session_state.get("active_project_id")
if active_project_id not in project_names:
    active_project_id = None

with st.sidebar:
    st.markdown("## ✳️ DatasetTest Overview")
    st.caption("TURN DATA INTO INTELLIGENCE")
    st.divider()
    st.caption("Akses anonim · tidak perlu login")
    uploaded = st.file_uploader(
        "Upload dataset (opsional)",
        type=["csv", "xlsx"],
        key="dataset_uploader",
    )
    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        if uploaded.size > 50 * 1024 * 1024:
            st.error("Ukuran file melebihi batas 50 MB.")
            st.stop()
        try:
            data_token = (uploaded.name, hashlib.sha256(uploaded_bytes).hexdigest())
            if st.session_state.get("dataset_token") != data_token:
                imported_data = load_uploaded_data(uploaded_bytes, uploaded.name)
                validate_dataset(imported_data)
                st.session_state.dataset_original = imported_data.copy()
                st.session_state.dataset_active = imported_data.copy()
                st.session_state.dataset_token = data_token
                st.session_state.dataset_source = uploaded.name
                st.session_state.target_column = imported_data.columns[-1]
                st.session_state.last_transform = ""
        except (UnicodeDecodeError, pd.errors.ParserError, OSError, ValueError, ImportError) as exc:
            st.error(f"Dataset tidak bisa dibaca: {exc}")
            st.stop()
    else:
        if st.session_state.get("dataset_token") != "demo":
            demo_data = make_sample_data()
            st.session_state.dataset_original = demo_data.copy()
            st.session_state.dataset_active = demo_data.copy()
            st.session_state.dataset_token = "demo"
            st.session_state.dataset_source = "Data sintetis · demo"
            st.session_state.target_column = "Churn"
            st.session_state.last_transform = ""

    data = st.session_state.dataset_active.copy()
    original_data = st.session_state.dataset_original.copy()
    if st.session_state.target_column not in data.columns:
        st.session_state.target_column = data.columns[-1]
    target = st.selectbox(
        "Pilih kolom target",
        options=data.columns.tolist(),
        key="target_column",
    )
    source_name = st.session_state.dataset_source

    st.markdown("#### WORKSPACE")
    page = st.radio(
        "Navigasi utama",
        [
            "Overview Dashboard",
            "Data Explorer",
            "Data Preparation",
            "EDA Studio",
            "Statistics Lab",
            "Model Studio",
            "My Workspace",
        ],
        label_visibility="collapsed",
        key="main_navigation",
    )
    global_search = st.text_input(
        "⌕ Cari di dataset",
        placeholder="Cari nilai di seluruh kolom...",
        help="Pencarian diterapkan pada tabel Data Explorer.",
    )
    st.markdown("#### DATASET AKTIF")
    st.caption(f"📄 {source_name}")
    st.caption(f"{len(data):,} baris · {data.shape[1]} kolom")
    project_options: list[str | None] = [None, *project_names]
    st.session_state.active_project_id = st.selectbox(
        "Proyek aktif",
        options=project_options,
        index=project_options.index(active_project_id),
        format_func=lambda project_id: (
            "Tanpa proyek aktif" if project_id is None else project_names[project_id]
        ),
    )
    theme_mode = st.selectbox(
        "Tema aplikasi",
        ["Dark", "Light", "System"],
        key="theme_mode",
    )
    st.divider()
    threshold = st.slider("Ambang churn", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    test_size = st.slider("Proporsi data uji", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    folds = st.slider("Jumlah fold CV", min_value=2, max_value=8, value=5)
    st.divider()
    st.caption("Workspace ini hanya berlaku untuk sesi saat ini.")

if theme_mode == "Light":
    st.markdown(
        """
        <style>
        .stApp { background: #f4f6fb; color: #172033; }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #dfe4ef; }
        [data-testid="stSidebar"] *, h1, h2, h3 { color: #172033 !important; }
        p, label, [data-testid="stMarkdownContainer"] { color: #34415b; }
        [data-testid="stMetric"] { background: #fff; border-color: #dfe4ef; }
        [data-testid="stMetricValue"] { color: #172033 !important; }
        .hero { background: linear-gradient(115deg, #fff, #eaf5f3); border-color: #dfe4ef; }
        .hero-title { color: #172033; }
        .hero-copy { color: #52617b; }
        .topbar, .quick-card, div[data-testid="stPlotlyChart"] {
            background: #fff; border-color: #dfe4ef;
        }
        .topbar-brand { color: #172033; }
        .topbar-meta, .section-label { color: #68758d; }
        [data-testid="stDataFrame"] { border-color: #dfe4ef; }
        </style>
        """,
        unsafe_allow_html=True,
    )
elif theme_mode == "System":
    st.markdown(
        """
        <style>
        @media (prefers-color-scheme: light) {
            .stApp { background: #f4f6fb; color: #172033; }
            [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #dfe4ef; }
            [data-testid="stSidebar"] *, h1, h2, h3 { color: #172033 !important; }
            p, label, [data-testid="stMarkdownContainer"] { color: #34415b; }
            [data-testid="stMetric"] { background: #fff; border-color: #dfe4ef; }
            [data-testid="stMetricValue"] { color: #172033 !important; }
            .hero { background: linear-gradient(115deg, #fff, #eaf5f3); border-color: #dfe4ef; }
            .hero-title { color: #172033; }
            .hero-copy { color: #52617b; }
            .topbar, .quick-card, div[data-testid="stPlotlyChart"] {
                background: #fff; border-color: #dfe4ef;
            }
            .topbar-brand { color: #172033; }
            .topbar-meta, .section-label { color: #68758d; }
            [data-testid="stDataFrame"] { border-color: #dfe4ef; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

if page == "Model Studio":
    try:
        model_results, fitted_models, prediction_results = evaluate_models(
            data=data,
            target=target,
            test_size=test_size,
            folds=folds,
            threshold=threshold,
        )
    except (ValueError, TypeError) as exc:
        st.error(f"Tidak dapat menjalankan evaluasi: {exc}")
        st.stop()

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-kicker">✳ &nbsp; STATISTICS • DATA ANALYTICS • MACHINE LEARNING</div>
        <div class="hero-title">DatasetTest <span>Overview</span></div>
        <p class="hero-copy">Turn Data Into Intelligence. Eksplorasi dataset dan temukan insight dari data aktual.</p>
        <p class="hero-developer">Developer: Daffa Rikza Mansuri</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="topbar">
        <div><span class="topbar-brand">DatasetTest Overview</span>
        <span class="topbar-meta"> &nbsp; / &nbsp; {page}</span></div>
        <div class="topbar-meta">● Dataset aktif &nbsp; {html.escape(source_name)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Overview Dashboard":
    chart_dataset_id = hashlib.sha256(
        repr(st.session_state.dataset_token).encode("utf-8")
    ).hexdigest()[:12]
    missing_count = int(data.isna().sum().sum())
    duplicate_count = int(data.duplicated().sum())
    quality = 100 * (1 - missing_count / max(data.size, 1))

    st.markdown('<div class="section-label">01 / Dataset pulse</div>', unsafe_allow_html=True)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Total records", f"{len(data):,}")
    metric_cols[1].metric("Total features", f"{data.shape[1] - 1:,}")
    metric_cols[2].metric("Missing values", f"{missing_count:,}")
    metric_cols[3].metric("Duplicate records", f"{duplicate_count:,}")
    metric_cols[4].metric("Data completeness", f"{quality:.1f}%")
    st.caption("Kelengkapan = 100% dikurangi proporsi sel kosong; dihitung dari dataset aktif.")

    st.markdown('<div class="section-label">02 / Isi dataset aktif</div>', unsafe_allow_html=True)
    dataset_info_cols = st.columns([1.5, 1])
    with dataset_info_cols[0]:
        st.subheader("Cuplikan data")
        st.caption(f"Sumber: {source_name}")
        st.dataframe(data.head(10), use_container_width=True, hide_index=True)
    with dataset_info_cols[1]:
        st.subheader("Profil kolom")
        column_profile = pd.DataFrame(
            {
                "Kolom": data.columns,
                "Tipe": [str(data[column].dtype) for column in data.columns],
                "Kosong": [int(data[column].isna().sum()) for column in data.columns],
                "Unik": [int(data[column].nunique(dropna=True)) for column in data.columns],
            }
        )
        st.dataframe(column_profile, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">03 / At a glance</div>', unsafe_allow_html=True)
    chart_cols = st.columns([1.1, 1.9])
    with chart_cols[0]:
        st.subheader("Komposisi target")
        composition_tabs = st.tabs(["Ringkasan", "Line chart", "Candlestick"])
        with composition_tabs[0]:
            distribution = (
                data[target].dropna().astype(str).value_counts()
                .rename_axis("Kelas").reset_index(name="Jumlah")
            )
            st.plotly_chart(
                px.pie(
                    distribution,
                    names="Kelas",
                    values="Jumlah",
                    hole=0.68,
                    color_discrete_sequence=["#c9f36a", "#a78bfa", "#67e8f9", "#fb7185"],
                ).update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#d7def1",
                    legend_title_text="",
                ),
                use_container_width=True,
            )

        with composition_tabs[1]:
            group_columns = [column for column in data.columns if column != target]
            if not group_columns:
                st.info("Tambahkan setidaknya satu fitur selain kolom target untuk membuat line chart.")
            else:
                group_column = st.selectbox(
                    "Kelompokkan komposisi berdasarkan",
                    group_columns,
                    key=f"target_composition_line_group_{chart_dataset_id}",
                )
                line_data = data[[group_column, target]].dropna().copy()
                if pd.api.types.is_numeric_dtype(line_data[group_column]):
                    unique_values = line_data[group_column].nunique()
                    if unique_values < 2:
                        st.info("Fitur yang dipilih perlu memiliki setidaknya dua nilai berbeda.")
                    else:
                        bin_count = min(10, unique_values)
                        line_data["_group"] = pd.qcut(
                            line_data[group_column],
                            q=bin_count,
                            duplicates="drop",
                        )
                else:
                    line_data["_group"] = line_data[group_column].astype(str)

                if "_group" in line_data.columns and not line_data.empty:
                    line_data["_class"] = line_data[target].astype(str)
                    line_counts = (
                        line_data.groupby(["_group", "_class"], observed=True, sort=True)
                        .size()
                        .rename("Jumlah")
                        .reset_index()
                    )
                    line_counts["Proporsi (%)"] = (
                        line_counts["Jumlah"]
                        / line_counts.groupby("_group")["Jumlah"].transform("sum")
                        * 100
                    )
                    group_order = line_counts["_group"].astype(str).drop_duplicates().tolist()
                    line_counts["_group"] = line_counts["_group"].astype(str)
                    figure = px.line(
                        line_counts,
                        x="_group",
                        y="Proporsi (%)",
                        color="_class",
                        markers=True,
                        category_orders={"_group": group_order},
                        color_discrete_sequence=["#c9f36a", "#a78bfa", "#67e8f9", "#fb7185"],
                        labels={"_group": group_column, "_class": "Kelas"},
                    )
                    figure.update_layout(
                        margin=dict(t=10, b=10, l=10, r=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#d7def1",
                        legend_title_text="",
                    )
                    st.plotly_chart(figure, use_container_width=True)
                    if pd.api.types.is_numeric_dtype(line_data[group_column]):
                        st.caption("Nilai fitur numerik dibagi ke dalam hingga 10 kuantil.")
                elif line_data.empty:
                    st.info("Tidak ada baris lengkap untuk fitur pengelompokan dan target yang dipilih.")

        with composition_tabs[2]:
            st.caption("Grafik candlestick hanya menggunakan kolom OHLC aktual dari dataset.")
            numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
            normalized_columns = {
                "".join(character for character in str(column).lower() if character.isalnum()): column
                for column in numeric_columns
            }

            def matching_ohlc_column(name: str) -> str | None:
                return next(
                    (
                        normalized_columns[candidate]
                        for candidate in (name, f"{name}price")
                        if candidate in normalized_columns
                    ),
                    None,
                )

            detected_ohlc = {
                field: matching_ohlc_column(field.lower())
                for field in ("Open", "High", "Low", "Close")
            }
            detected_complete_ohlc = (
                all(column is not None for column in detected_ohlc.values())
                and len(set(detected_ohlc.values())) == 4
            )
            has_confirmed_ohlc = st.checkbox(
                "Kolom yang dipilih berisi data OHLC aktual",
                value=detected_complete_ohlc,
                disabled=len(numeric_columns) < 4,
                key=f"candlestick_confirm_ohlc_{chart_dataset_id}",
            )
            ohlc_columns = st.columns(4)
            selected_ohlc: dict[str, str] = {}
            for column, field in zip(ohlc_columns, ("Open", "High", "Low", "Close")):
                default_column = detected_ohlc[field]
                options = ["Pilih kolom", *numeric_columns]
                default_option = default_column or "Pilih kolom"
                with column:
                    selected_ohlc[field] = st.selectbox(
                        field,
                        options,
                        index=options.index(default_option),
                        disabled=not has_confirmed_ohlc,
                        key=f"candlestick_{field.lower()}_{chart_dataset_id}",
                    )

            if len(numeric_columns) < 4:
                st.info("Candlestick memerlukan empat kolom numerik: Open, High, Low, dan Close.")
            elif not has_confirmed_ohlc:
                st.info(
                    "Kolom OHLC tidak terdeteksi otomatis. Konfirmasi data OHLC aktual "
                    "untuk memetakan nama kolom secara manual."
                )
            elif "Pilih kolom" in selected_ohlc.values():
                st.info("Pilih kolom Open, High, Low, dan Close untuk menampilkan candlestick.")
            elif len(set(selected_ohlc.values())) != 4:
                st.warning("Pilih empat kolom numerik yang berbeda untuk Open, High, Low, dan Close.")
            else:
                date_columns = [
                    column
                    for column in data.columns
                    if not pd.api.types.is_numeric_dtype(data[column])
                ]
                date_options: list[str | None] = [None, *date_columns]
                date_column = st.selectbox(
                    "Kolom tanggal (opsional)",
                    date_options,
                    format_func=lambda column: "Urutan baris" if column is None else str(column),
                    key=f"candlestick_date_{chart_dataset_id}",
                )
                candlestick_data = data[list(selected_ohlc.values())].copy()
                candlestick_data.columns = list(selected_ohlc)
                candlestick_data = candlestick_data.apply(pd.to_numeric, errors="coerce")
                valid_rows = candlestick_data.notna().all(axis=1)
                valid_rows &= candlestick_data["High"] >= candlestick_data[
                    ["Open", "Close", "Low"]
                ].max(axis=1)
                valid_rows &= candlestick_data["Low"] <= candlestick_data[
                    ["Open", "Close", "High"]
                ].min(axis=1)
                candlestick_data = candlestick_data.loc[valid_rows]
                valid_ohlc_count = len(candlestick_data)
                if date_column is None:
                    candle_x = candlestick_data.index
                else:
                    parsed_dates = pd.to_datetime(data.loc[candlestick_data.index, date_column], errors="coerce")
                    date_valid = parsed_dates.notna()
                    candlestick_data = candlestick_data.loc[date_valid]
                    candle_x = parsed_dates.loc[date_valid]
                    date_order = candle_x.sort_values().index
                    candlestick_data = candlestick_data.loc[date_order]
                    candle_x = candle_x.loc[date_order]

                if candlestick_data.empty:
                    st.warning("Tidak ditemukan baris OHLC valid untuk kolom yang dipilih.")
                else:
                    if valid_ohlc_count < len(data) or len(candlestick_data) < valid_ohlc_count:
                        st.caption("Baris dengan nilai OHLC atau tanggal tidak valid dikecualikan.")
                    candle = go.Figure(
                        data=[
                            go.Candlestick(
                                x=candle_x,
                                open=candlestick_data["Open"],
                                high=candlestick_data["High"],
                                low=candlestick_data["Low"],
                                close=candlestick_data["Close"],
                                increasing_line_color="#c9f36a",
                                decreasing_line_color="#fb7185",
                            )
                        ]
                    )
                    candle.update_layout(
                        margin=dict(t=10, b=10, l=10, r=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#d7def1",
                        xaxis_rangeslider_visible=False,
                        showlegend=False,
                    )
                    st.plotly_chart(candle, use_container_width=True)
    with chart_cols[1]:
        st.subheader("Preview fitur numerik")
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        if numeric_columns:
            numeric_summary = data[numeric_columns].describe().T.reset_index(names="Fitur")
            st.dataframe(numeric_summary, use_container_width=True, hide_index=True)
        else:
            st.info("Dataset ini tidak memiliki fitur numerik untuk diringkas.")

    quick_cols = st.columns(3)
    quick_links = [
        ("🔎 Data Explorer", "Cari, filter, dan unduh dataset aktif."),
        ("📈 EDA Studio", "Bangun grafik langsung dari kolom dataset."),
        ("🧠 Model Studio", "Lihat evaluasi model klasifikasi yang tersedia."),
    ]
    for column, (title, description) in zip(quick_cols, quick_links):
        with column:
            st.markdown(
                f'<div class="quick-card"><strong>{title}</strong><br><span>{description}</span></div>',
                unsafe_allow_html=True,
            )

elif page == "Data Explorer":
    st.markdown('<div class="section-label">01 / Dataset workspace</div>', unsafe_allow_html=True)
    missing_by_column = data.isna().sum()
    explorer_cols = st.columns(4)
    explorer_cols[0].metric("Rows", f"{len(data):,}")
    explorer_cols[1].metric("Columns", f"{data.shape[1]:,}")
    explorer_cols[2].metric("Missing cells", f"{int(missing_by_column.sum()):,}")
    explorer_cols[3].metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")

    st.subheader("Preview dataset")
    view_data = data
    if global_search.strip():
        search_match = data.astype("string").apply(
            lambda column: column.str.contains(global_search.strip(), case=False, na=False)
        )
        view_data = data.loc[search_match.any(axis=1)]
        st.caption(f"{len(view_data):,} baris cocok dengan pencarian.")
    st.dataframe(view_data, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download dataset aktif (CSV)",
        data=data.to_csv(index=False).encode("utf-8-sig"),
        file_name="statlytics_dataset.csv",
        mime="text/csv",
    )
    workbook_buffer = BytesIO()
    with pd.ExcelWriter(workbook_buffer, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Dataset", index=False)
        pd.DataFrame(
            {
                "Kolom": data.columns,
                "Tipe data": [str(data[column].dtype) for column in data.columns],
                "Nilai kosong": [int(data[column].isna().sum()) for column in data.columns],
                "Nilai unik": [int(data[column].nunique(dropna=True)) for column in data.columns],
            }
        ).to_excel(writer, sheet_name="Profil kolom", index=False)
    st.download_button(
        "⬇️ Download dataset aktif (Excel)",
        data=workbook_buffer.getvalue(),
        file_name="statlytics_dataset.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("Informasi kolom")
    column_info = pd.DataFrame(
        {
            "Kolom": data.columns,
            "Tipe data": [str(data[column].dtype) for column in data.columns],
            "Nilai kosong": [int(data[column].isna().sum()) for column in data.columns],
            "Nilai unik": [int(data[column].nunique(dropna=True)) for column in data.columns],
        }
    )
    st.dataframe(column_info, use_container_width=True, hide_index=True)

elif page == "My Workspace":
    st.markdown('<div class="section-label">01 / Personal projects</div>', unsafe_allow_html=True)
    st.caption(
        "Proyek dan aktivitas tersimpan di SQLite lokal, terpisah untuk setiap akun. "
        "File dataset belum disalin otomatis ke proyek."
    )
    with st.form("create_project_form", clear_on_submit=True):
        project_name_input = st.text_input("Nama proyek baru", max_chars=80)
        create_submitted = st.form_submit_button("＋ Create project", type="primary")
    if create_submitted:
        try:
            new_project_id = create_project(guest_user, project_name_input)
            st.session_state.active_project_id = new_project_id
            st.session_state.workspace_notice = "Proyek berhasil dibuat."
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    if st.session_state.get("workspace_notice"):
        st.success(st.session_state.workspace_notice)
        st.session_state.pop("workspace_notice", None)

    user_projects = list_projects(guest_user)
    project_ids = [str(project["project_id"]) for project in user_projects]
    st.subheader("My Projects")
    if user_projects:
        project_table = pd.DataFrame(
            [
                {
                    "Proyek": project["name"],
                    "Favorite": "★" if project["is_favorite"] else "",
                    "Dibuat": project["created_at"],
                    "Diperbarui": project["updated_at"],
                }
                for project in user_projects
            ]
        )
        st.dataframe(project_table, use_container_width=True, hide_index=True)

        if st.session_state.get("workspace_project_id") not in project_ids:
            st.session_state.workspace_project_id = project_ids[0]
        selected_project_id = st.selectbox(
            "Pilih proyek untuk dikelola",
            options=project_ids,
            format_func=lambda project_id: project_names.get(
                project_id,
                next(
                    (
                        str(project["name"])
                        for project in user_projects
                        if str(project["project_id"]) == project_id
                    ),
                    project_id,
                ),
            ),
            key="workspace_project_id",
        )
        selected_project = next(
            project for project in user_projects
            if str(project["project_id"]) == selected_project_id
        )
        rename_columns = st.columns(2)
        with rename_columns[0]:
            with st.form(f"rename_project_{selected_project_id}"):
                new_project_name = st.text_input(
                    "Ubah nama proyek",
                    value=str(selected_project["name"]),
                    max_chars=80,
                    key=f"rename_name_{selected_project_id}_{selected_project['updated_at']}",
                )
                rename_submitted = st.form_submit_button("Rename")
            if rename_submitted:
                try:
                    if rename_project(
                        guest_user,
                        selected_project_id,
                        new_project_name,
                    ):
                        st.session_state.workspace_notice = "Nama proyek diperbarui."
                        st.rerun()
                    st.error("Proyek tidak ditemukan di workspace akun ini.")
                except ValueError as exc:
                    st.error(str(exc))
        with rename_columns[1]:
            favorite_label = (
                "★ Remove favorite"
                if selected_project["is_favorite"]
                else "☆ Add favorite"
            )
            if st.button(favorite_label):
                set_project_favorite(
                    guest_user,
                    selected_project_id,
                    not bool(selected_project["is_favorite"]),
                )
                st.rerun()
            confirm_delete = st.checkbox("Saya yakin ingin menghapus proyek ini")
            if st.button("Delete project", disabled=not confirm_delete):
                if delete_project(guest_user, selected_project_id):
                    st.session_state.pop("workspace_project_id", None)
                    if st.session_state.get("active_project_id") == selected_project_id:
                        st.session_state.active_project_id = None
                    st.session_state.workspace_notice = "Proyek dihapus."
                    st.rerun()
                st.error("Proyek tidak ditemukan di workspace akun ini.")
    else:
        st.info("Belum ada proyek. Buat proyek pertama untuk memulai workspace.")

    st.markdown('<div class="section-label">02 / Recent activity</div>', unsafe_allow_html=True)
    activities = list_activities(guest_user)
    if activities:
        st.dataframe(
            pd.DataFrame(activities).rename(
                columns={
                    "action": "Aktivitas",
                    "detail": "Detail",
                    "created_at": "Waktu (UTC)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Aktivitas proyek akan muncul di sini.")

elif page == "Data Preparation":
    st.markdown('<div class="section-label">01 / Data quality check</div>', unsafe_allow_html=True)
    if st.session_state.get("last_transform"):
        st.success(f"Proses terakhir: {st.session_state.last_transform}")
    prep_metrics = st.columns(4)
    prep_metrics[0].metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
    prep_metrics[1].metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")
    prep_metrics[2].metric("Rows", f"{len(data):,}")
    prep_metrics[3].metric("Columns", f"{data.shape[1]:,}")
    st.caption("Dataset asli tetap tersimpan; perubahan diterapkan hanya ke dataset aktif.")

    st.markdown('<div class="section-label">02 / Apply transformation</div>', unsafe_allow_html=True)
    operation = st.selectbox(
        "Pilih proses",
        [
            "Isi missing values (median/mode)",
            "Hapus baris dengan missing values",
            "Hapus baris duplikat",
            "Batasi outlier dengan IQR",
            "Standardisasi fitur numerik",
            "One-hot encode fitur kategori",
        ],
    )
    feature_columns = [column for column in data.columns if column != target]
    numeric_features = data[feature_columns].select_dtypes(include=np.number).columns.tolist()
    categorical_features = [
        column for column in feature_columns if column not in numeric_features
    ]
    selected_feature: str | None = None
    if operation == "Batasi outlier dengan IQR" and numeric_features:
        selected_feature = st.selectbox("Fitur numerik", numeric_features)
    elif operation == "One-hot encode fitur kategori" and categorical_features:
        selected_feature = st.selectbox("Fitur kategori", categorical_features)
    elif operation in ["Batasi outlier dengan IQR", "One-hot encode fitur kategori"]:
        st.warning("Tidak ada kolom yang sesuai untuk operasi ini.")

    if st.button("Terapkan ke dataset aktif", type="primary"):
        transformed = data.copy()
        if operation == "Isi missing values (median/mode)":
            for column in feature_columns:
                if transformed[column].isna().any():
                    if pd.api.types.is_numeric_dtype(transformed[column]):
                        fill_value = transformed[column].median()
                    else:
                        modes = transformed[column].mode(dropna=True)
                        if modes.empty:
                            continue
                        fill_value = modes.iloc[0]
                    transformed[column] = transformed[column].fillna(fill_value)
        elif operation == "Hapus baris dengan missing values":
            transformed = transformed.dropna().reset_index(drop=True)
        elif operation == "Hapus baris duplikat":
            transformed = transformed.drop_duplicates().reset_index(drop=True)
        elif operation == "Batasi outlier dengan IQR" and selected_feature is not None:
            values = transformed[selected_feature]
            q1, q3 = values.quantile([0.25, 0.75])
            spread = q3 - q1
            transformed[selected_feature] = values.clip(
                lower=q1 - 1.5 * spread,
                upper=q3 + 1.5 * spread,
            )
        elif operation == "Standardisasi fitur numerik":
            for column in numeric_features:
                values = transformed[column]
                mean = values.mean()
                std = values.std(ddof=0)
                if pd.notna(std) and std > 0:
                    transformed[column] = (values - mean) / std
        elif operation == "One-hot encode fitur kategori" and selected_feature is not None:
            transformed = pd.get_dummies(
                transformed,
                columns=[selected_feature],
                prefix=selected_feature,
                dtype=int,
            )
        st.session_state.dataset_active = transformed
        st.session_state.last_transform = operation
        st.rerun()

    compare_columns = st.columns(2)
    with compare_columns[0]:
        st.subheader("Asli")
        st.caption(f"{len(original_data):,} baris · {original_data.shape[1]} kolom")
        st.dataframe(original_data.head(8), use_container_width=True, hide_index=True)
    with compare_columns[1]:
        st.subheader("Aktif")
        st.caption(f"{len(data):,} baris · {data.shape[1]} kolom")
        st.dataframe(data.head(8), use_container_width=True, hide_index=True)
    if st.button("Kembalikan ke dataset asli"):
        st.session_state.dataset_active = original_data.copy()
        st.session_state.last_transform = ""
        st.rerun()

elif page == "Statistics Lab":
    st.markdown('<div class="section-label">01 / Descriptive statistics</div>', unsafe_allow_html=True)
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = data.select_dtypes(exclude=np.number).columns.tolist()
    if numeric_columns:
        descriptive = data[numeric_columns].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).T
        descriptive["Median"] = data[numeric_columns].median()
        descriptive["Mode"] = [
            data[column].mode(dropna=True).iloc[0]
            if not data[column].mode(dropna=True).empty
            else np.nan
            for column in numeric_columns
        ]
        descriptive["Variance"] = data[numeric_columns].var()
        descriptive["Skewness"] = data[numeric_columns].skew()
        descriptive["Kurtosis"] = data[numeric_columns].kurtosis()
        st.dataframe(descriptive.reset_index(names="Fitur"), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-label">02 / Correlation explorer</div>', unsafe_allow_html=True)
        if len(numeric_columns) >= 2:
            corr_controls = st.columns([1, 2])
            with corr_controls[0]:
                corr_method = st.selectbox(
                    "Metode korelasi",
                    ["pearson", "spearman", "kendall"],
                )
                selected_numeric = st.multiselect(
                    "Pilih fitur numerik",
                    numeric_columns,
                    default=numeric_columns[: min(8, len(numeric_columns))],
                )
            if len(selected_numeric) >= 2:
                correlation = data[selected_numeric].corr(method=corr_method)
                with corr_controls[1]:
                    st.plotly_chart(
                        px.imshow(
                            correlation,
                            text_auto=".2f",
                            zmin=-1,
                            zmax=1,
                            color_continuous_scale=["#a78bfa", "#111a2f", "#c9f36a"],
                            aspect="auto",
                        ).update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#d7def1",
                            coloraxis_colorbar_title="r",
                        ),
                        use_container_width=True,
                    )
                st.dataframe(correlation, use_container_width=True)
            else:
                st.info("Pilih setidaknya dua fitur untuk menghitung korelasi.")
        else:
            st.info("Korelasi membutuhkan setidaknya dua kolom numerik.")
    else:
        st.info("Dataset aktif tidak memiliki kolom numerik untuk statistik deskriptif.")

    st.markdown('<div class="section-label">03 / Inferential tests</div>', unsafe_allow_html=True)
    test_name = st.selectbox(
        "Uji statistik",
        [
            "Shapiro-Wilk (normalitas)",
            "Independent t-test (Welch)",
            "Mann-Whitney U",
            "One-way ANOVA",
            "Chi-square (independensi)",
        ],
    )
    if test_name == "Shapiro-Wilk (normalitas)":
        if not numeric_columns:
            st.info("Uji ini membutuhkan fitur numerik.")
        else:
            test_column = st.selectbox("Fitur numerik", numeric_columns, key="shapiro_column")
            if st.button("Jalankan Shapiro-Wilk"):
                sample = data[test_column].dropna()
                if len(sample) < 3:
                    st.error("Diperlukan minimal 3 nilai non-kosong.")
                else:
                    if len(sample) > 5000:
                        sample = sample.sample(5000, random_state=42)
                    result = stats.shapiro(sample)
                    st.metric("Statistic", f"{result.statistic:.4f}")
                    st.metric("P-value", f"{result.pvalue:.4g}")
                    decision = "Tolak H₀: data berbeda signifikan dari distribusi normal." if result.pvalue < 0.05 else "Belum cukup bukti untuk menolak H₀ normalitas."
                    st.write(f"**H₀:** sampel berasal dari distribusi normal. **α = 0,05.** {decision}")
    elif test_name == "Chi-square (independensi)":
        if len(categorical_columns) < 2:
            st.info("Uji chi-square membutuhkan dua kolom kategori.")
        else:
            test_cols = st.columns(2)
            with test_cols[0]:
                cat_a = st.selectbox("Variabel kategori A", categorical_columns, key="chi_a")
            with test_cols[1]:
                cat_b = st.selectbox(
                    "Variabel kategori B",
                    [column for column in categorical_columns if column != cat_a],
                    key="chi_b",
                )
            if st.button("Jalankan Chi-square"):
                contingency = pd.crosstab(data[cat_a], data[cat_b])
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    st.error("Setiap variabel harus memiliki setidaknya dua kategori.")
                else:
                    result = stats.chi2_contingency(contingency)
                    st.metric("Statistic", f"{result.statistic:.4f}")
                    st.metric("P-value", f"{result.pvalue:.4g}")
                    decision = "Ada bukti asosiasi antara variabel." if result.pvalue < 0.05 else "Belum ada bukti asosiasi yang signifikan."
                    st.write(f"**H₀:** kedua variabel independen. **α = 0,05.** {decision}")
    else:
        if not numeric_columns or not categorical_columns:
            st.info("Pilih dataset dengan fitur numerik dan kategori untuk uji perbandingan.")
        else:
            test_cols = st.columns(2)
            with test_cols[0]:
                value_column = st.selectbox("Fitur numerik", numeric_columns, key="comparison_value")
                group_column = st.selectbox("Kolom grup", categorical_columns, key="comparison_group")
            groups = data[group_column].dropna().unique().tolist()
            if len(groups) < 2:
                st.info("Kolom grup harus memiliki setidaknya dua nilai berbeda.")
            elif test_name in ["Independent t-test (Welch)", "Mann-Whitney U"]:
                chosen_groups = st.multiselect(
                    "Pilih tepat dua grup",
                    groups,
                    default=groups[:2],
                    max_selections=2,
                )
                if st.button("Jalankan uji perbandingan"):
                    if len(chosen_groups) != 2:
                        st.error("Pilih tepat dua grup.")
                    else:
                        samples = [
                            data.loc[data[group_column] == group, value_column].dropna().to_numpy()
                            for group in chosen_groups
                        ]
                        if min(map(len, samples)) < 2:
                            st.error("Setiap grup harus memiliki minimal dua nilai.")
                        else:
                            if test_name == "Independent t-test (Welch)":
                                result = stats.ttest_ind(*samples, equal_var=False)
                                h0 = "Rata-rata kedua grup sama."
                            else:
                                result = stats.mannwhitneyu(*samples, alternative="two-sided")
                                h0 = "Distribusi kedua grup sama."
                            st.metric("Statistic", f"{result.statistic:.4f}")
                            st.metric("P-value", f"{result.pvalue:.4g}")
                            decision = "Tolak H₀ pada α = 0,05." if result.pvalue < 0.05 else "Belum cukup bukti untuk menolak H₀ pada α = 0,05."
                            st.write(f"**H₀:** {h0} {decision}")
            else:
                st.caption("ANOVA membandingkan rerata numerik antar seluruh grup.")
                if st.button("Jalankan One-way ANOVA"):
                    samples = [
                        data.loc[data[group_column] == group, value_column].dropna().to_numpy()
                        for group in groups
                    ]
                    samples = [sample for sample in samples if len(sample) >= 2]
                    if len(samples) < 2:
                        st.error("Diperlukan minimal dua grup dengan masing-masing dua nilai.")
                    else:
                        result = stats.f_oneway(*samples)
                        st.metric("Statistic", f"{result.statistic:.4f}")
                        st.metric("P-value", f"{result.pvalue:.4g}")
                        decision = "Setidaknya ada satu rerata grup yang berbeda." if result.pvalue < 0.05 else "Belum ada bukti perbedaan rerata yang signifikan."
                        st.write(f"**H₀:** semua rerata grup sama. **α = 0,05.** {decision}")

elif page == "EDA Studio":
    st.markdown('<div class="section-label">01 / Interactive chart builder</div>', unsafe_allow_html=True)
    st.subheader("Eksplorasi data aktual")
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = data.select_dtypes(exclude=np.number).columns.tolist()
    chart_types = ["Histogram", "Bar chart"]
    if numeric_columns:
        chart_types.extend(["Boxplot", "Violin plot", "Scatter plot", "Line chart"])
    chart_type = st.selectbox(
        "Jenis visualisasi",
        chart_types,
    )
    chart_controls = st.columns(2)
    with chart_controls[0]:
        x_column = st.selectbox("Kolom X", options=data.columns.tolist())
    with chart_controls[1]:
        if chart_type in ["Scatter plot", "Line chart", "Boxplot", "Violin plot"]:
            y_options = numeric_columns
            y_column = st.selectbox("Kolom Y", options=y_options)
        else:
            y_column = None

    if chart_type == "Histogram":
        if x_column not in numeric_columns:
            st.warning("Pilih kolom numerik untuk histogram.")
        else:
            figure = px.histogram(data, x=x_column, color=target, color_discrete_sequence=["#c9f36a", "#a78bfa", "#67e8f9"])
            st.plotly_chart(figure, use_container_width=True)
    elif chart_type == "Bar chart":
        counts = data[x_column].astype("string").fillna("(kosong)").value_counts().rename_axis(x_column).reset_index(name="Jumlah")
        st.plotly_chart(px.bar(counts, x=x_column, y="Jumlah", color_discrete_sequence=["#a78bfa"]), use_container_width=True)
    elif chart_type == "Boxplot":
        st.plotly_chart(px.box(data, x=x_column, y=y_column, color=target, color_discrete_sequence=["#c9f36a", "#a78bfa", "#67e8f9"]), use_container_width=True)
    elif chart_type == "Violin plot":
        st.plotly_chart(px.violin(data, x=x_column, y=y_column, color=target, box=True, points="outliers"), use_container_width=True)
    elif chart_type == "Scatter plot":
        color_column = st.selectbox("Warna berdasarkan", options=[None, *categorical_columns], format_func=lambda value: value or "Tidak ada")
        st.plotly_chart(px.scatter(data, x=x_column, y=y_column, color=color_column), use_container_width=True)
    else:
        if x_column not in numeric_columns:
            st.warning("Pilih kolom numerik sebagai sumbu X untuk line chart.")
        else:
            st.plotly_chart(px.line(data.sort_values(x_column), x=x_column, y=y_column), use_container_width=True)

elif page == "Model Studio":
    st.markdown('<div class="section-label">01 / Model performance</div>', unsafe_allow_html=True)
    st.subheader("Evaluasi model klasifikasi")
    st.dataframe(
        model_results.style.format(
            {
                "Accuracy": "{:.1%}",
                "Precision": "{:.1%}",
                "Recall": "{:.1%}",
                "F1-score": "{:.1%}",
                "ROC-AUC": "{:.1%}",
                "CV Accuracy": "{:.1%}",
                "CV Std": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    model_name = st.selectbox("Pilih model untuk prediksi", options=list(fitted_models))
    selected_predictions = prediction_results[["Baris data", "Aktual", model_name]].rename(
        columns={model_name: "Prediksi"}
    )
    selected_predictions["Status"] = np.where(
        selected_predictions["Aktual"] == selected_predictions["Prediksi"],
        "Cocok",
        "Belum cocok",
    )
    st.download_button(
        "⬇️ Download hasil prediksi CSV",
        data=selected_predictions.rename(columns={"Baris data": "Nomor baris"}).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"hasil_prediksi_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
    st.dataframe(selected_predictions.head(15), use_container_width=True, hide_index=True)
