"""
Student Academic Performance Predictor v2.0 - Interactive Streamlit Dashboard
-----------------------------------------------------------------------------
Interactive ML decision support interface featuring:
- 20 raw academic, behavioural & contextual inputs + 8 leak-free engineered features
- Live model switcher across 9 trained architectures (saved_models/registry/)
- Dynamic 'What-If' intervention simulator across key levers
- Global permutation importance & holdout benchmark tournament diagnostics
- Batch CSV cohort prediction & Hugging Face Pre-Trained Hub search/benchmark
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from src.explainability import simulate_what_if_trajectory
from src.dataset import DEFAULT_STUDENT_PROFILE, load_data
from src.features import engineer_features
from src.pretrained_hub import (
    PRETRAINED_CATALOG,
    search_huggingface_models,
    search_huggingface_datasets,
    benchmark_pretrained_models,
    predict_with_shalan1,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "saved_models"
REGISTRY_DIR = MODELS_DIR / "registry"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"


st.set_page_config(
    page_title="Student Academic Performance Predictor v2.0",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_pipeline_by_slug(slug: str | None = None):
    if slug:
        path = REGISTRY_DIR / f"{slug}.joblib"
        if path.exists():
            return joblib.load(path)
    best_path = MODELS_DIR / "best_model.joblib"
    if not best_path.exists():
        st.error(f"Trained model not found at '{best_path}'. Please run 'python train.py' first.")
        st.stop()
    return joblib.load(best_path)


@st.cache_data
def load_benchmark_report():
    path = REPORTS_DIR / "benchmark_results.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tier_info(score: float):
    if score >= 90.0:
        return "Grade A", "Dean's Honors / Outstanding Mastery", "🌟"
    elif score >= 80.0:
        return "Grade B", "Very Good / Strong Conceptual Grasp", "📘"
    elif score >= 70.0:
        return "Grade C", "Good / Satisfactory Academic Standing", "📗"
    elif score >= 60.0:
        return "Grade D", "Passing / Needs Academic Reinforcement", "⚠️"
    else:
        return "Grade F", "Academic Risk / Intensive Intervention Required", "🚨"


report = load_benchmark_report()
available_models = report.get("models", [])
slug_options = {m["name"]: m["slug"] for m in available_models} if available_models else {"Champion Model": None}

st.sidebar.title("🎓 Student Profile (v2.0)")
selected_model_name = st.sidebar.selectbox("Active ML Model Engine", list(slug_options.keys()), index=0)
selected_slug = slug_options[selected_model_name]
pipeline = load_pipeline_by_slug(selected_slug)

with st.sidebar.expander("📚 Academic & Assessment", expanded=True):
    previous_score = st.slider("Prior Exam Score", 0.0, 100.0, float(DEFAULT_STUDENT_PROFILE["previous_score"]), 1.0)
    quiz_average = st.slider("Quiz Average", 0.0, 100.0, float(DEFAULT_STUDENT_PROFILE["quiz_average"]), 1.0)
    assignment_completion_rate = st.slider("Assignment Completion (%)", 0.0, 100.0, float(DEFAULT_STUDENT_PROFILE["assignment_completion_rate"]), 1.0)
    hours_studied = st.slider("Weekly Study Hours", 1.0, 35.0, float(DEFAULT_STUDENT_PROFILE["hours_studied"]), 0.5)
    attendance_percent = st.slider("Class Attendance (%)", 40.0, 100.0, float(DEFAULT_STUDENT_PROFILE["attendance_percent"]), 1.0)
    practice_papers_solved = st.slider("Practice Papers Solved", 0, 20, int(DEFAULT_STUDENT_PROFILE["practice_papers_solved"]), 1)

with st.sidebar.expander("🧠 Wellbeing & Study Habits", expanded=False):
    sleep_hours = st.slider("Sleep (hrs/night)", 4.0, 11.0, float(DEFAULT_STUDENT_PROFILE["sleep_hours"]), 0.5)
    tutoring_sessions = st.slider("Tutoring Sessions / Month", 0, 12, int(DEFAULT_STUDENT_PROFILE["tutoring_sessions"]), 1)
    physical_activity_hours = st.slider("Physical Activity (hrs/wk)", 0.0, 20.0, float(DEFAULT_STUDENT_PROFILE["physical_activity_hours"]), 0.5)
    screen_time_hours = st.slider("Screen Time (hrs/day)", 0.0, 12.0, float(DEFAULT_STUDENT_PROFILE["screen_time_hours"]), 0.5)
    stress_level = st.slider("Stress Level (1-10)", 1, 10, int(DEFAULT_STUDENT_PROFILE["stress_level"]), 1)
    commute_minutes = st.slider("Daily Commute (mins)", 0, 120, int(DEFAULT_STUDENT_PROFILE["commute_minutes"]), 5)

with st.sidebar.expander("🏫 Context & Demographics", expanded=False):
    parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
    motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
    access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], index=1)
    parental_education = st.selectbox("Parental Education", ["High School", "Bachelor", "Master+"], index=1)
    extracurricular = st.radio("Extracurriculars", ["Yes", "No"], index=0, horizontal=True)
    internet_access = st.radio("Home Broadband", ["Yes", "No"], index=0, horizontal=True)
    school_type = st.radio("School Type", ["Public", "Private"], index=0, horizontal=True)
    learning_style = st.selectbox("Learning Style", ["Visual", "Auditory", "Kinesthetic", "Reading/Writing"], index=0)

input_student = {
    "hours_studied": float(hours_studied),
    "attendance_percent": float(attendance_percent),
    "previous_score": float(previous_score),
    "quiz_average": float(quiz_average),
    "assignment_completion_rate": float(assignment_completion_rate),
    "practice_papers_solved": float(practice_papers_solved),
    "sleep_hours": float(sleep_hours),
    "tutoring_sessions": int(tutoring_sessions),
    "physical_activity_hours": float(physical_activity_hours),
    "screen_time_hours": float(screen_time_hours),
    "stress_level": float(stress_level),
    "commute_minutes": float(commute_minutes),
    "parental_involvement": parental_involvement,
    "motivation_level": motivation_level,
    "access_to_resources": access_to_resources,
    "parental_education": parental_education,
    "extracurricular": extracurricular,
    "internet_access": internet_access,
    "school_type": school_type,
    "learning_style": learning_style,
}

input_df = pd.DataFrame([input_student])
predicted_score = round(float(np.clip(pipeline.predict(input_df)[0], 0.0, 100.0)), 1)
eng_df = engineer_features(input_df)
tier_name, tier_desc, icon = get_tier_info(predicted_score)

st.title("🎓 Student Academic Performance Studio (v2.0)")
st.markdown("Real-time multi-model ML prediction across 20 raw + 8 engineered features & Hugging Face Pre-Trained Hub.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Live Forecaster",
    "🔮 What-If Simulator",
    "📊 Benchmarks & Explainability",
    "📂 Cohort & Batch Scoring",
    "🤗 Pre-Trained Hub",
])

with tab1:
    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        st.metric(
            label=f"Projected Final Exam Score ({selected_model_name})",
            value=f"{predicted_score:.1f} / 100",
            delta=f"{predicted_score - previous_score:+.1f} vs prior baseline",
        )
        st.success(f"{icon} **{tier_name}** — {tier_desc}")
    with c2:
        st.metric("Academic Momentum", f"{eng_df['academic_momentum'].iloc[0]:.1f} pts")
        st.metric("Study Efficiency", f"{eng_df['study_efficiency'].iloc[0]:.3f}")
    with c3:
        st.metric("Engagement Score", f"{eng_df['engagement_score'].iloc[0]:.3f}")
        st.metric("Wellbeing Index", f"{eng_df['wellbeing_index'].iloc[0]:.3f}")

    st.subheader("Engineered Composite Features")
    eng_cols = [
        "study_efficiency", "academic_momentum", "sleep_deficit", "engagement_score",
        "study_sleep_balance", "digital_distraction_ratio", "formative_delta", "wellbeing_index",
    ]
    st.dataframe(eng_df[eng_cols].round(3), use_container_width=True)

with tab2:
    st.subheader("🔮 What-If Intervention Trajectory")
    lever_options = {
        "hours_studied": ("Weekly Study Hours", 1.0, 30.0),
        "attendance_percent": ("Attendance Rate (%)", 50.0, 100.0),
        "quiz_average": ("Quiz Average", 40.0, 100.0),
        "assignment_completion_rate": ("Assignment Completion (%)", 40.0, 100.0),
        "sleep_hours": ("Sleep Hours/Night", 4.0, 10.5),
        "stress_level": ("Stress Level (1-10)", 1.0, 10.0),
        "screen_time_hours": ("Screen Time Hours/Day", 0.5, 10.0),
    }
    var_col = st.selectbox("Select Intervention Lever", list(lever_options.keys()), format_func=lambda k: lever_options[k][0])
    x_label, lo, hi = lever_options[var_col]
    vals = np.linspace(lo, hi, 21).tolist()
    traj_df = simulate_what_if_trajectory(pipeline, input_student, var_col, vals)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(traj_df[var_col], traj_df["predicted_score"], color="#4f46e5", linewidth=2.5, marker="o")
    ax.scatter([input_student[var_col]], [predicted_score], color="#e11d48", s=110, zorder=5, label=f"Current ({input_student[var_col]} -> {predicted_score:.1f})")
    ax.axhline(90, color="#16a34a", linestyle="--", alpha=0.5, label="Grade A (90)")
    ax.axhline(80, color="#2563eb", linestyle="--", alpha=0.5, label="Grade B (80)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Predicted Final Score")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.subheader("📊 9-Model Benchmark Tournament & Permutation Importances")
    if available_models:
        st.dataframe(pd.DataFrame(available_models), use_container_width=True)
        imp = report.get("feature_importances", [])
        if imp:
            st.bar_chart(pd.DataFrame(imp).set_index("feature")["importance_mean"].head(12))
    else:
        st.info("Run `python train.py` to populate `reports/benchmark_results.json`.")

with tab4:
    st.subheader("📂 Cohort Overview & Batch CSV Scoring")
    uploaded = st.file_uploader("Upload student cohort CSV (with feature columns)", type=["csv"])
    if uploaded is not None:
        df_batch = pd.read_csv(uploaded)
        for col, default_val in DEFAULT_STUDENT_PROFILE.items():
            if col not in df_batch.columns:
                df_batch[col] = default_val
        df_batch["predicted_score"] = np.round(np.clip(pipeline.predict(df_batch), 0.0, 100.0), 1)
        st.dataframe(df_batch.head(50), use_container_width=True)
        st.download_button("Download Scored CSV", df_batch.to_csv(index=False), "scored_cohort.csv", "text/csv")
    elif (DATA_DIR / "student_data.csv").exists():
        sample_df = load_data(DATA_DIR / "student_data.csv").head(20)
        st.caption(f"Previewing first 20 of {report.get('dataset_summary', {}).get('total_rows', 20000):,} cohort rows:")
        st.dataframe(sample_df, use_container_width=True)

with tab5:
    st.subheader("🤗 Hugging Face Pre-Trained Model & Dataset Hub")
    col_hf1, col_hf2 = st.columns([2, 1])
    with col_hf1:
        query = st.text_input("Search Hugging Face Hub", value="student performance")
    with col_hf2:
        search_kind = st.radio("Resource Type", ["models", "datasets"], horizontal=True)

    if st.button("Search Hub"):
        results = search_huggingface_models(query) if search_kind == "models" else search_huggingface_datasets(query)
        st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.markdown("#### Curated Community Pre-Trained Models")
    st.dataframe(pd.DataFrame(PRETRAINED_CATALOG), use_container_width=True)

    c_p1, c_p2 = st.columns(2)
    with c_p1:
        if st.button("Run Pre-Trained Benchmark (Shalan1 & SergeyR256)"):
            with st.spinner("Evaluating Hugging Face models on real UCI & 10K datasets..."):
                st.dataframe(pd.DataFrame(benchmark_pretrained_models(DATA_DIR / "external")), use_container_width=True)
    with c_p2:
        if st.button("Predict Current Student with HF Shalan1 RF"):
            rf_score = predict_with_shalan1(
                hours_studied, previous_score, extracurricular, sleep_hours, practice_papers_solved
            )
            st.info(f"Shalan1/student-performance-predictor-models Prediction: **{rf_score:.1f} / 100** (vs Local Model: **{predicted_score:.1f} / 100**)")
