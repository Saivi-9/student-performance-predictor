"""
FastAPI Academic Performance Prediction REST API Application (v2.0).

Provides real-time inference across 9 trained ML models, 20 raw + 8 engineered features,
local per-student feature attribution, batch cohort scoring, intervention simulations,
live benchmark telemetry, and Hugging Face Hub pre-trained model search & comparison.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    StudentInput,
    PredictionResponse,
    FactorContribution,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchPredictionItem,
    SimulationResponse,
    SimulationStep,
)
from src.dataset import DEFAULT_STUDENT_PROFILE, load_data
from src.features import engineer_features
from src.explainability import simulate_what_if_trajectory
from src.pretrained_hub import (
    PRETRAINED_CATALOG,
    search_huggingface_models,
    search_huggingface_datasets,
    benchmark_pretrained_models,
    predict_with_shalan1,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "saved_models"
REGISTRY_DIR = MODELS_DIR / "registry"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"

app = FastAPI(
    title="Student Academic Performance Predictor API",
    description="Enterprise ML Studio REST API with 9-model tournament, 28 features, and Hugging Face Pre-trained Hub.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline_cache: Dict[str, Any] = {}


def get_pipeline(model_slug: Optional[str] = None):
    """Load and cache the requested model pipeline (or champion by default)."""
    key = model_slug or "best_model"
    if key in _pipeline_cache:
        return _pipeline_cache[key], key

    if model_slug:
        candidate_path = REGISTRY_DIR / f"{model_slug}.joblib"
        if not candidate_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_slug}' not found in registry.",
            )
        model_path = candidate_path
    else:
        model_path = MODELS_DIR / "best_model.joblib"
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Trained model not found at {model_path}. Run train.py first.",
            )

    loaded = joblib.load(model_path)
    _pipeline_cache[key] = loaded
    return loaded, key


def _read_benchmark_report() -> Dict[str, Any]:
    report_path = REPORTS_DIR / "benchmark_results.json"
    if not report_path.exists():
        return {}
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


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


ATTRIBUTION_FEATURES = [
    ("previous_score", "Prior Exam Score"),
    ("quiz_average", "Quiz Average"),
    ("hours_studied", "Weekly Study Hours"),
    ("attendance_percent", "Class Attendance"),
    ("assignment_completion_rate", "Assignment Completion"),
    ("practice_papers_solved", "Practice Papers"),
    ("sleep_hours", "Nightly Sleep"),
    ("tutoring_sessions", "Tutoring Sessions"),
    ("stress_level", "Stress Level"),
    ("screen_time_hours", "Recreational Screen Time"),
    ("parental_involvement", "Parental Involvement"),
    ("motivation_level", "Motivation Level"),
]


def _compute_local_contributions(pipeline, data_dict: Dict[str, Any]) -> List[FactorContribution]:
    """Estimate per-feature point contribution relative to the cohort average profile."""
    baseline = DEFAULT_STUDENT_PROFILE.copy()
    # Batch all perturbations in one DataFrame call for sub-5ms speed
    rows = [baseline]
    perturbed_meta = []
    for feat, label in ATTRIBUTION_FEATURES:
        row = baseline.copy()
        row[feat] = data_dict.get(feat, baseline[feat])
        rows.append(row)
        perturbed_meta.append((feat, label))

    batch_df = pd.DataFrame(rows)
    preds = pipeline.predict(batch_df)
    base_pred = float(preds[0])

    contributions: List[FactorContribution] = []
    for idx, (feat, label) in enumerate(perturbed_meta, start=1):
        delta = round(float(preds[idx]) - base_pred, 2)
        contributions.append(
            FactorContribution(
                feature=feat,
                label=label,
                impact=delta,
                direction="positive" if delta >= 0 else "negative",
            )
        )

    contributions.sort(key=lambda c: abs(c.impact), reverse=True)
    return contributions[:8]


@app.get("/health", tags=["System"])
def health_check():
    pipeline_loaded = False
    try:
        get_pipeline()
        pipeline_loaded = True
    except Exception:
        pass

    report = _read_benchmark_report()
    return {
        "status": "HEALTHY",
        "service": "Student Academic Performance Predictor",
        "version": "2.0.0",
        "model_loaded": pipeline_loaded,
        "champion_model": report.get("champion_model", "Champion Pipeline"),
        "dataset_rows": report.get("dataset_summary", {}).get("total_rows", 20000),
        "total_features": report.get("dataset_summary", {}).get("total_pipeline_features", 28),
        "models_available": len(report.get("models", [])),
    }


@app.get("/v1/models", tags=["Models"])
def list_models():
    """Return all trained models in the tournament registry with live metrics."""
    report = _read_benchmark_report()
    return {
        "champion_model": report.get("champion_model"),
        "champion_slug": report.get("champion_slug"),
        "models": report.get("models", []),
    }


@app.post("/v1/predict", response_model=PredictionResponse, tags=["Forecasting"])
def predict_performance(
    student: StudentInput,
    model_slug: Optional[str] = Query(default=None, description="Optional model slug from /v1/models"),
):
    pipeline, resolved_key = get_pipeline(model_slug)
    data_dict = student.model_dump()
    df_row = pd.DataFrame([data_dict])

    pred = float(pipeline.predict(df_row)[0])
    bounded_pred = round(max(0.0, min(100.0, pred)), 1)
    tier_name, tier_label, icon = get_tier_details(bounded_pred)

    # Confidence interval derived from model test RMSE
    report = _read_benchmark_report()
    rmse = float(report.get("test_metrics", {}).get("RMSE", 2.5))
    if model_slug:
        for m in report.get("models", []):
            if m.get("slug") == model_slug:
                rmse = float(m.get("test_rmse", rmse))
                break

    ci_low = round(max(0.0, bounded_pred - 1.28 * rmse), 1)
    ci_high = round(min(100.0, bounded_pred + 1.28 * rmse), 1)

    # Compute engineered features live
    eng_df = engineer_features(df_row)
    engineered_dict = {
        k: round(float(eng_df[k].iloc[0]), 3)
        for k in [
            "study_efficiency",
            "academic_momentum",
            "sleep_deficit",
            "engagement_score",
            "study_sleep_balance",
            "digital_distraction_ratio",
            "formative_delta",
            "wellbeing_index",
        ]
    }

    contributions = _compute_local_contributions(pipeline, data_dict)
    effort_density = round(student.hours_studied / (student.tutoring_sessions + 1), 2)
    delta_prev = round(bounded_pred - student.previous_score, 1)

    strengths = []
    if student.previous_score >= 75:
        strengths.append(f"Strong prior exam baseline ({student.previous_score:.0f} pts)")
    if student.quiz_average >= student.previous_score:
        strengths.append(f"Positive formative momentum (+{student.quiz_average - student.previous_score:.1f} pts quiz gain)")
    if student.attendance_percent >= 85:
        strengths.append(f"Consistent lecture attendance ({student.attendance_percent:.0f}%)")
    if student.assignment_completion_rate >= 85:
        strengths.append(f"High coursework completion ({student.assignment_completion_rate:.0f}%)")
    if not strengths:
        strengths.append(f"Current academic anchor at {student.previous_score:.0f} pts prior baseline")

    recommendations = []
    if student.hours_studied < 10.0:
        recommendations.append("Increasing weekly study by +3 hours yields strong marginal score gains.")
    if student.sleep_hours < 7.0 or student.sleep_hours > 9.5:
        recommendations.append("Optimize nightly sleep toward 7.5-8.5 hrs to minimize cognitive sleep deficit.")
    if student.attendance_percent < 82.0:
        recommendations.append("Attendance below 82% creates syllabus coverage gaps; target 90%+.")
    if student.stress_level >= 7.0:
        recommendations.append("Elevated stress index (>=7/10); incorporate physical activity or pacing breaks.")
    if student.screen_time_hours >= 5.0:
        recommendations.append("High recreational screen time (>5 hrs/day) correlates with lower study retention.")
    if student.assignment_completion_rate < 80.0:
        recommendations.append("Boosting assignment completion above 85% unlocks attendance-coursework synergy.")

    model_display = report.get("champion_model", "Champion Pipeline")
    if model_slug:
        for m in report.get("models", []):
            if m.get("slug") == model_slug:
                model_display = m.get("name", model_slug)
                break

    return PredictionResponse(
        predicted_score=bounded_pred,
        confidence_interval=[ci_low, ci_high],
        model_used=model_display,
        grade_tier=tier_name,
        tier_label=tier_label,
        status_icon=icon,
        delta_vs_previous=delta_prev,
        effort_density_hrs_per_session=effort_density,
        engineered_features=engineered_dict,
        feature_contributions=contributions,
        behavioral_diagnostics={
            "strengths": strengths,
            "recommendations": recommendations or ["Current academic and wellbeing habits are well-balanced."],
        },
        input_features=data_dict,
    )


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Forecasting"])
def predict_batch(payload: BatchPredictionRequest):
    """Run real-time batch scoring across an entire cohort of student records."""
    if not payload.students:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Student list cannot be empty.")

    pipeline, key = get_pipeline(payload.model_slug)
    rows = [s.model_dump() for s in payload.students]
    df_batch = pd.DataFrame(rows)
    eng_batch = engineer_features(df_batch)
    preds = np.clip(pipeline.predict(df_batch), 0.0, 100.0)

    items: List[BatchPredictionItem] = []
    at_risk = 0
    for idx, p in enumerate(preds):
        score = round(float(p), 1)
        tier_name, _, _ = get_tier_details(score)
        if score < 65.0:
            at_risk += 1
        items.append(
            BatchPredictionItem(
                index=idx + 1,
                predicted_score=score,
                grade_tier=tier_name,
                academic_momentum=round(float(eng_batch["academic_momentum"].iloc[idx]), 1),
                engagement_score=round(float(eng_batch["engagement_score"].iloc[idx]), 3),
            )
        )

    report = _read_benchmark_report()
    model_display = report.get("champion_model", key)
    if payload.model_slug:
        for m in report.get("models", []):
            if m.get("slug") == payload.model_slug:
                model_display = m.get("name", payload.model_slug)
                break

    return BatchPredictionResponse(
        count=len(items),
        model_used=model_display,
        average_score=round(float(np.mean(preds)), 1),
        at_risk_count=at_risk,
        predictions=items,
    )


@app.post("/v1/simulate", response_model=SimulationResponse, tags=["Sensitivity"])
def simulate_intervention(
    student: StudentInput,
    feature_name: str = "hours_studied",
    min_val: float = 1.0,
    max_val: float = 25.0,
    steps: int = 15,
    model_slug: Optional[str] = Query(default=None),
):
    pipeline, key = get_pipeline(model_slug)
    data_dict = student.model_dump()

    if feature_name not in data_dict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown feature: {feature_name}",
        )

    vals = np.linspace(min_val, max_val, steps).tolist()
    traj_df = simulate_what_if_trajectory(pipeline, data_dict, feature_name, vals)

    sim_steps = [
        SimulationStep(value=round(float(row[feature_name]), 2), predicted_score=round(float(row["predicted_score"]), 1))
        for _, row in traj_df.iterrows()
    ]

    curr_val = float(data_dict[feature_name])
    curr_score = round(float(np.clip(pipeline.predict(pd.DataFrame([data_dict]))[0], 0.0, 100.0)), 1)

    report = _read_benchmark_report()
    model_display = report.get("champion_model", key)
    if model_slug:
        for m in report.get("models", []):
            if m.get("slug") == model_slug:
                model_display = m.get("name", model_slug)
                break

    return SimulationResponse(
        feature_name=feature_name,
        model_used=model_display,
        trajectory=sim_steps,
        current_value=curr_val,
        current_score=curr_score,
    )


@app.get("/v1/benchmarks", tags=["Analytics"])
def get_benchmarks():
    """Return real computed benchmark tournament results, feature importances, and residual samples."""
    report = _read_benchmark_report()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmark report not found. Run 'python train.py' first.",
        )
    return report


@app.get("/v1/dataset/summary", tags=["Data"])
def get_dataset_summary(sample_limit: int = Query(default=12, ge=1, le=50)):
    """Return live stats and sample records from the primary 20k dataset and external UCI benchmark."""
    primary_path = DATA_DIR / "student_data.csv"
    uci_path = DATA_DIR / "external" / "uci_student_performance.csv"
    hf10k_path = DATA_DIR / "external" / "hf_student_performance_10k.csv"

    report = _read_benchmark_report()
    samples = []
    if primary_path.exists():
        df_head = pd.read_csv(primary_path, nrows=sample_limit).fillna("N/A")
        samples = df_head.to_dict(orient="records")

    external_sources = []
    if uci_path.exists():
        df_uci = pd.read_csv(uci_path)
        external_sources.append({
            "name": "UCI Student Performance (Math + Portuguese)",
            "source": "Hugging Face (DElmazi/Student_Performance)",
            "rows": int(len(df_uci)),
            "columns": int(df_uci.shape[1]),
        })
    if hf10k_path.exists():
        df_10k = pd.read_csv(hf10k_path)
        external_sources.append({
            "name": "10K Student Performance Index",
            "source": "Hugging Face (imranx654/Student-Performance-Predictor)",
            "rows": int(len(df_10k)),
            "columns": int(df_10k.shape[1]),
        })

    return {
        "dataset_summary": report.get("dataset_summary", {}),
        "sample_records": samples,
        "external_sources": external_sources,
    }


@app.get("/v1/pretrained/catalog", tags=["Hugging Face Hub"])
def get_pretrained_catalog():
    """Return curated community pretrained models on Hugging Face Hub."""
    return {"catalog": PRETRAINED_CATALOG}


@app.get("/v1/pretrained/search", tags=["Hugging Face Hub"])
def search_pretrained_hub(
    query: str = Query(default="student performance", min_length=2),
    kind: str = Query(default="models", pattern="^(models|datasets)$"),
    limit: int = Query(default=10, ge=1, le=25),
):
    """Live-search Hugging Face Hub for pretrained models or datasets."""
    try:
        if kind == "datasets":
            items = search_huggingface_datasets(query=query, limit=limit)
        else:
            items = search_huggingface_models(query=query, limit=limit)
        return {"query": query, "kind": kind, "count": len(items), "results": items}
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"HF Hub query failed: {exc}")


@app.post("/v1/pretrained/benchmark", tags=["Hugging Face Hub"])
def run_pretrained_benchmark(refresh: bool = Query(default=False)):
    """Download and evaluate real Hugging Face pretrained models on benchmark datasets."""
    cache_path = REPORTS_DIR / "pretrained_benchmarks.json"
    if cache_path.exists() and not refresh:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    results = benchmark_pretrained_models(DATA_DIR / "external")
    payload = {"benchmarks": results}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


@app.post("/v1/pretrained/predict", tags=["Hugging Face Hub"])
def run_pretrained_prediction(student: StudentInput):
    """Run live inference with the community Hugging Face Shalan1 RandomForest model."""
    try:
        pred = predict_with_shalan1(
            hours_studied=student.hours_studied,
            previous_score=student.previous_score,
            extracurricular=student.extracurricular,
            sleep_hours=student.sleep_hours,
            practice_papers_solved=student.practice_papers_solved,
        )
        tier_name, tier_label, icon = get_tier_details(pred)
        return {
            "repo_id": "Shalan1/student-performance-predictor-models",
            "predicted_score": pred,
            "grade_tier": tier_name,
            "tier_label": tier_label,
            "status_icon": icon,
        }
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Pretrained inference failed: {exc}")


@app.get("/v1/system/registry-diagnostics", tags=["System"])
def get_registry_diagnostics():
    """Audit all trained model artifacts in the registry and dataset files on disk."""
    artifacts = []
    if REGISTRY_DIR.exists():
        for p in sorted(REGISTRY_DIR.glob("*.joblib")):
            artifacts.append({
                "slug": p.stem,
                "path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "size_kb": round(p.stat().st_size / 1024.0, 2),
                "available": True,
            })

    champion_exists = (MODELS_DIR / "best_model.joblib").exists()
    datasets = {}
    for name, path in [
        ("primary_20k", DATA_DIR / "student_data.csv"),
        ("uci_benchmark", DATA_DIR / "external" / "uci_student_performance.csv"),
        ("hf_10k_benchmark", DATA_DIR / "external" / "hf_student_performance_10k.csv"),
    ]:
        datasets[name] = {
            "available": path.exists(),
            "size_kb": round(path.stat().st_size / 1024.0, 2) if path.exists() else 0.0,
        }

    return {
        "status": "READY" if champion_exists and len(artifacts) > 0 else "DEGRADED",
        "champion_artifact_available": champion_exists,
        "registry_count": len(artifacts),
        "registry_artifacts": artifacts,
        "datasets": datasets,
    }


# Mount Production React Frontend if built
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))
if os.path.exists(frontend_dist):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
