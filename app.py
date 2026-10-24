"""
Student Academic Performance Predictor - Interactive Streamlit Dashboard
-----------------------------------------------------------------------
An interactive decision support interface for educators, academic advisors, and students:
- Live score forecasting with grade tier classifications
- Dynamic 'What-If' simulator for student intervention planning
- Model explainability via permutation feature importance
- Cohort benchmarks and residual diagnostic analysis
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from src.explainability import compute_permutation_importance, plot_feature_importance, simulate_what_if_trajectory
from src.dataset import load_data, split_data


st.set_page_config(
    page_title="Student Academic Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        margin-bottom: 15px;
    }
    .badge-grade-a {
        background-color: #dcfce7;
        color: #15803d;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-grade-b {
        background-color: #dbeafe;
        color: #1d4ed8;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-grade-c {
        background-color: #fef9c3;
        color: #a16207;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-grade-d {
        background-color: #ffedd5;
        color: #c2410c;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-grade-f {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_pipeline():
    """Load the trained model pipeline."""
    model_path = Path(__file__).resolve().parent / "saved_models" / "best_model.joblib"
    if not model_path.exists():
        st.error(f"Trained model not found at '{model_path}'. Please run 'python train.py' first.")
        st.stop()
    return joblib.load(model_path)


@st.cache_data
def get_validation_data():
    """Load dataset and split to extract validation set for explainability."""
    data_path = Path(__file__).resolve().parent / "data" / "student_data.csv"
    if not data_path.exists():
        return None, None
    df = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, random_state=42)
    return X_val, y_val


def get_tier_info(score: float):
    if score >= 90.0:
        return "Grade A", "badge-grade-a", "Dean's Honors / Outstanding Mastery", "🌟"
    elif score >= 80.0:
        return "Grade B", "badge-grade-b", "Very Good / Strong Conceptual Grasp", "📘"
    elif score >= 70.0:
        return "Grade C", "badge-grade-c", "Good / Satisfactory Academic Standing", "📗"
    elif score >= 60.0:
        return "Grade D", "badge-grade-d", "Passing / Needs Academic Reinforcement", "⚠️"
    else:
        return "Grade F", "badge-grade-f", "Academic Risk / Intensive Intervention Required", "🚨"


# Sidebar Navigation & Student Profile Inputs
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
st.sidebar.title("Student Profile")
st.sidebar.markdown("Configure the student's academic and behavioral indicators:")

hours_studied = st.sidebar.slider(
    "Weekly Study Hours",
    min_value=1.0,
    max_value=25.0,
    value=6.5,
    step=0.5,
    help="Average number of hours dedicated to self-study each week."
)

attendance_percent = st.sidebar.slider(
    "Class Attendance Rate (%)",
    min_value=40.0,
    max_value=100.0,
    value=88.0,
    step=1.0,
    help="Percentage of lectures and lab sessions attended."
)

previous_score = st.sidebar.slider(
    "Prior Exam Score (0 - 100)",
    min_value=30.0,
    max_value=100.0,
    value=76.0,
    step=1.0,
    help="Numeric score on the preceding benchmark examination."
)

sleep_hours = st.sidebar.slider(
    "Average Sleep (Hours/Night)",
    min_value=4.0,
    max_value=11.0,
    value=7.5,
    step=0.5,
    help="Average nightly sleep duration."
)

tutoring_sessions = st.sidebar.slider(
    "Tutoring Sessions / Month",
    min_value=0,
    max_value=10,
    value=2,
    step=1,
    help="Number of structured one-on-one or group tutoring sessions attended."
)

col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    parental = st.selectbox(
        "Parental Support",
        options=["Low", "Medium", "High"],
        index=1,
        help="Reported level of family engagement and educational support."
    )
with col_p2:
    internet = st.selectbox(
        "Home Internet",
        options=["Yes", "No"],
        index=0,
        help="Reliable high-speed broadband access at home."
    )

extracurricular = st.sidebar.radio(
    "Extracurricular Activities",
    options=["Yes", "No"],
    index=0,
    horizontal=True,
    help="Participation in varsity athletics, arts, clubs, or community service."
)

input_student = {
    "hours_studied": float(hours_studied),
    "attendance_percent": float(attendance_percent),
    "previous_score": float(previous_score),
    "sleep_hours": float(sleep_hours),
    "tutoring_sessions": int(tutoring_sessions),
    "parental_involvement": parental,
    "extracurricular": extracurricular,
    "internet_access": internet
}

pipeline = load_trained_pipeline()
input_df = pd.DataFrame([input_student])
raw_score = float(pipeline.predict(input_df)[0])
predicted_score = round(max(0.0, min(100.0, raw_score)), 1)
tier_name, badge_class, tier_desc, icon = get_tier_info(predicted_score)

# Main Dashboard Interface
st.title("🎓 Student Academic Performance Predictor")
st.markdown("Automated predictive modeling and decision support for academic achievement and student success.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Live Score Forecasting",
    "🔮 'What-If' Intervention Studio",
    "📊 Explainability & Feature Importance",
    "📈 Benchmark & Model Diagnostics"
])

# ----------------- TAB 1: FORECASTING -----------------
with tab1:
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #64748b;">PROJECTED FINAL EXAM SCORE</h4>
            <div style="font-size: 3.2rem; font-weight: 800; color: #0f172a; margin: 10px 0;">
                {predicted_score} <span style="font-size: 1.4rem; color: #94a3b8;">/ 100</span>
            </div>
            <div class="{badge_class}">
                {icon} {tier_name}: {tier_desc}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        delta_score = round(predicted_score - previous_score, 1)
        st.metric(
            label="Trajectory Delta vs Previous Exam",
            value=f"{predicted_score} pts",
            delta=f"{delta_score:+} pts",
            help="Difference between predicted final score and prior baseline exam."
        )
        st.metric(
            label="Study Effort Density",
            value=f"{hours_studied / (tutoring_sessions + 1):.1f} hrs/session",
            help="Weekly study hours relative to tutoring sessions."
        )

    with col3:
        sleep_dev = abs(sleep_hours - 8.0)
        sleep_status = "Optimal" if sleep_dev <= 0.5 else ("Minor Deficit" if sleep_dev <= 1.5 else "Suboptimal")
        st.metric(
            label="Sleep Regularity",
            value=f"{sleep_hours} hrs/night",
            delta=sleep_status,
            delta_color="normal" if sleep_dev <= 0.5 else "off"
        )
        attendance_risk = "Low Risk" if attendance_percent >= 85 else ("Moderate Risk" if attendance_percent >= 75 else "High Absence Risk")
        st.metric(
            label="Attendance Safety",
            value=f"{attendance_percent:.0f}%",
            delta=attendance_risk,
            delta_color="normal" if attendance_percent >= 80 else "inverse"
        )

    st.subheader("💡 Behavioral Diagnostics & Recommendations")
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.info(f"""
        **Strengths Identified:**
        - **Prior Foundation:** Prior score of {previous_score:.0f} provides a solid academic base.
        - **Attendance:** {attendance_percent:.0f}% lecture presence ensures regular syllabus exposure.
        - **Resource Access:** High-speed internet access ({internet}) supports digital learning.
        """)
    with rec_col2:
        action_items = []
        if hours_studied < 8.0:
            action_items.append("Increasing weekly independent study by +2 hours could unlock higher grade tiers.")
        if sleep_hours < 7.0:
            action_items.append("Sleep duration is below the recommended 7.5 - 8.5h window; memory consolidation may be impaired.")
        if tutoring_sessions == 0:
            action_items.append("Enrolling in 1-2 structured tutoring sessions per month provides targeted remedial support.")
        if not action_items:
            action_items.append("Current learning habits are well-balanced; sustain current study pacing through finals.")

        st.warning("**Targeted Interventions:**\n" + "\n".join(f"- {item}" for item in action_items))

# ----------------- TAB 2: WHAT-IF SIMULATOR -----------------
with tab2:
    st.subheader("🔮 'What-If' Student Intervention Simulation")
    st.markdown("Simulate how targeted behavioural adjustments impact projected examination performance:")

    sim_lever = st.selectbox(
        "Select Intervention Lever to Simulate:",
        options=[
            "Weekly Study Hours (1 to 20 hrs)",
            "Class Attendance (60% to 100%)",
            "Tutoring Sessions (0 to 8 sessions)",
            "Prior Score Improvement (50 to 95)"
        ]
    )

    if "Weekly Study Hours" in sim_lever:
        var_col = "hours_studied"
        var_vals = [float(x) for x in np.linspace(1.0, 20.0, 20)]
        x_label = "Weekly Study Hours"
    elif "Class Attendance" in sim_lever:
        var_col = "attendance_percent"
        var_vals = [float(x) for x in np.linspace(60.0, 100.0, 21)]
        x_label = "Attendance Percentage (%)"
    elif "Tutoring Sessions" in sim_lever:
        var_col = "tutoring_sessions"
        var_vals = list(range(0, 9))
        x_label = "Tutoring Sessions Attended"
    else:
        var_col = "previous_score"
        var_vals = [float(x) for x in np.linspace(50.0, 100.0, 21)]
        x_label = "Prior Exam Baseline"

    traj_df = simulate_what_if_trajectory(pipeline, input_student, var_col, var_vals)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        traj_df[var_col],
        traj_df["predicted_score"],
        color="#2563eb",
        linewidth=2.5,
        marker="o",
        markersize=5,
        label="Projected Score Trajectory"
    )

    # Current student point
    current_val = input_student[var_col]
    ax.scatter(
        [current_val],
        [predicted_score],
        color="#dc2626",
        s=120,
        zorder=5,
        label=f"Current Baseline ({current_val}: {predicted_score:.1f} pts)"
    )

    # Threshold horizontal reference lines
    ax.axhline(90, color="#16a34a", linestyle="--", alpha=0.5, label="Grade A Threshold (90 pts)")
    ax.axhline(80, color="#2563eb", linestyle="--", alpha=0.5, label="Grade B Threshold (80 pts)")
    ax.axhline(70, color="#ca8a04", linestyle="--", alpha=0.5, label="Grade C Threshold (70 pts)")

    ax.set_xlabel(x_label, fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Final Exam Score", fontsize=11, fontweight="bold")
    ax.set_title(f"Sensitivity Analysis: {x_label} vs. Projected Final Score", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="lower right", fontsize=9)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

# ----------------- TAB 3: EXPLAINABILITY -----------------
with tab3:
    st.subheader("📊 Permutation Feature Importance & Model Transparency")
    st.markdown("""
    Permutation feature importance measures the increase in prediction error after shuffling each feature's values,
    breaking the relationship with the target. Features with the largest drops in $R^2$ are the most critical.
    """)

    X_val, y_val = get_validation_data()
    if X_val is not None:
        importance_df = compute_permutation_importance(pipeline, X_val, y_val, n_repeats=5, random_state=42)
        fig_imp = plot_feature_importance(importance_df, title="Global Permutation Feature Importance (Holdout Validation)")
        st.pyplot(fig_imp)
        plt.close(fig_imp)

        st.dataframe(importance_df, use_container_width=True)
    else:
        st.warning("Validation dataset not found. Please verify data/student_data.csv exists.")

# ----------------- TAB 4: BENCHMARKS -----------------
with tab4:
    st.subheader("📈 Model Benchmarks & Validation Artifacts")
    fig_dir = Path(__file__).resolve().parent / "reports" / "figures"

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("#### 1. Actual vs. Predicted Correlation")
        actual_fig = fig_dir / "actual_vs_predicted.png"
        if actual_fig.exists():
            st.image(str(actual_fig), caption="Champion Model Actual vs. Predicted on Holdout Test Set")
        else:
            st.info("Run `python train.py` to generate this diagnostic plot.")

    with col_b2:
        st.markdown("#### 2. Error Residual Homoscedasticity")
        residual_fig = fig_dir / "residual_plot.png"
        if residual_fig.exists():
            st.image(str(residual_fig), caption="Residual Distribution across Predicted Range")
        else:
            st.info("Run `python train.py` to generate this diagnostic plot.")

    st.markdown("#### 3. Model Benchmark Comparison")
    comp_fig = fig_dir / "model_comparison.png"
    if comp_fig.exists():
        st.image(str(comp_fig), caption="Cross-Validation Comparison: Linear vs Ridge vs Random Forest")
