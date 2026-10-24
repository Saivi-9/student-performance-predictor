"""
FastAPI Academic Performance Prediction REST API Application.
"""

from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import StudentInput, PredictionResponse, SimulationResponse, SimulationStep
from src.explainability import simulate_what_if_trajectory, compute_permutation_importance
from src.dataset import load_data, split_data


app = FastAPI(
    title="Student Academic Performance Predictor API",
    description="REST API for predicting exam outcomes, grade tiers, and intervention sensitivity.",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        model_path = Path(__file__).resolve().parent.parent.parent / "saved_models" / "best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
        _pipeline = joblib.load(model_path)
    return _pipeline


def get_tier_details(score: float):
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


@app.get("/health", tags=["System"])
def health_check():
    pipeline_loaded = False
    try:
        get_pipeline()
        pipeline_loaded = True
    except Exception:
        pass

    return {
        "status": "HEALTHY",
        "service": "Student Academic Performance Predictor",
        "version": "1.2.0",
        "model_loaded": pipeline_loaded
    }


@app.post("/v1/predict", response_model=PredictionResponse, tags=["Forecasting"])
def predict_performance(student: StudentInput):
    pipeline = get_pipeline()
    data_dict = student.model_dump()
    df_row = pd.DataFrame([data_dict])

    pred = float(pipeline.predict(df_row)[0])
    bounded_pred = round(max(0.0, min(100.0, pred)), 1)
    tier_name, tier_label, icon = get_tier_details(bounded_pred)

    effort_density = round(student.hours_studied / (student.tutoring_sessions + 1), 2)
    delta_prev = round(bounded_pred - student.previous_score, 1)

    # Behavioral diagnostics
    recommendations = []
    if student.hours_studied < 8.0:
        recommendations.append("Increasing weekly study by +2 hours unlocks potential for higher grade standing.")
    if student.sleep_hours < 7.0:
        recommendations.append("Sleep duration is below optimal (7.5-8.5 hrs); memory consolidation risk.")
    if student.attendance_percent < 80.0:
        recommendations.append("Attendance is below 80%; risk of syllabus coverage gaps.")

    return PredictionResponse(
        predicted_score=bounded_pred,
        grade_tier=tier_name,
        tier_label=tier_label,
        status_icon=icon,
        delta_vs_previous=delta_prev,
        effort_density_hrs_per_session=effort_density,
        behavioral_diagnostics={
            "strengths": [
                f"Prior exam baseline of {student.previous_score:.0f} pts",
                f"Attendance presence at {student.attendance_percent:.0f}%",
            ],
            "recommendations": recommendations or ["Current academic habits are well-balanced."]
        },
        input_features=data_dict
    )


@app.post("/v1/simulate", response_model=SimulationResponse, tags=["Sensitivity"])
def simulate_intervention(
    student: StudentInput,
    feature_name: str = "hours_studied",
    min_val: float = 1.0,
    max_val: float = 20.0,
    steps: int = 15
):
    pipeline = get_pipeline()
    data_dict = student.model_dump()

    if feature_name not in data_dict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown feature: {feature_name}"
        )

    vals = np.linspace(min_val, max_val, steps).tolist()
    traj_df = simulate_what_if_trajectory(pipeline, data_dict, feature_name, vals)

    sim_steps = [
        SimulationStep(value=round(float(row[feature_name]), 2), predicted_score=round(float(row["predicted_score"]), 1))
        for _, row in traj_df.iterrows()
    ]

    curr_val = float(data_dict[feature_name])
    df_curr = pd.DataFrame([data_dict])
    curr_score = round(float(pipeline.predict(df_curr)[0]), 1)

    return SimulationResponse(
        feature_name=feature_name,
        trajectory=sim_steps,
        current_value=curr_val,
        current_score=curr_score
    )


# Mount Production React Frontend if built
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))
if os.path.exists(frontend_dist):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
