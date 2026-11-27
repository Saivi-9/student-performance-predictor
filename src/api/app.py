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

    try:
        loaded = joblib.load(model_path)
    except ModuleNotFoundError:
        fallback_path = MODELS_DIR / "best_model.joblib"
        if not fallback_path.exists():
            raise
        loaded = joblib.load(fallback_path)
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


@app.post("/v1/predict/intervention-plan", tags=["Sensitivity"])
def generate_intervention_plan(
    student: StudentInput,
    target_score: float = Query(default=85.0, ge=50.0, le=100.0),
    model_slug: Optional[str] = Query(default=None),
):
    """Evaluate actionable academic levers and rank concrete interventions by projected score gain."""
    pipeline, _ = get_pipeline(model_slug)
    base_dict = student.model_dump()
    base_score = round(float(np.clip(pipeline.predict(pd.DataFrame([base_dict]))[0], 0.0, 100.0)), 1)

    candidate_levers = [
        ("hours_studied", "Increase weekly study time", min(25.0, round(student.hours_studied + 4.0, 1))),
        ("attendance_percent", "Raise lecture attendance", min(100.0, max(student.attendance_percent, 95.0))),
        ("assignment_completion_rate", "Boost assignment completion", min(100.0, max(student.assignment_completion_rate, 95.0))),
        ("practice_papers_solved", "Solve additional past papers", min(15, student.practice_papers_solved + 3)),
        ("tutoring_sessions", "Add weekly tutoring sessions", min(8, student.tutoring_sessions + 2)),
        ("sleep_hours", "Optimize nightly sleep schedule", 8.0),
    ]

    rows = []
    meta = []
    for feat, label, proposed_val in candidate_levers:
        row = base_dict.copy()
        row[feat] = proposed_val
        rows.append(row)
        meta.append((feat, label, base_dict[feat], proposed_val))

    # Also evaluate combined intervention across all levers
    combined_row = base_dict.copy()
    for feat, _, proposed_val in candidate_levers:
        combined_row[feat] = proposed_val
    rows.append(combined_row)

    preds = np.clip(pipeline.predict(pd.DataFrame(rows)), 0.0, 100.0)
    interventions = []
    for idx, (feat, label, curr_val, prop_val) in enumerate(meta):
        proj = round(float(preds[idx]), 1)
        gain = round(proj - base_score, 1)
        interventions.append({
            "feature": feat,
            "action": label,
            "current_value": curr_val,
            "recommended_value": prop_val,
            "projected_score": proj,
            "score_gain": gain,
            "meets_target": proj >= target_score,
        })

    interventions.sort(key=lambda item: item["score_gain"], reverse=True)
    combined_score = round(float(preds[-1]), 1)
    return {
        "current_score": base_score,
        "target_score": target_score,
        "combined_projected_score": combined_score,
        "combined_score_gain": round(combined_score - base_score, 1),
        "target_achievable": combined_score >= target_score,
        "ranked_interventions": interventions,
    }


@app.post("/v1/predict/model-consensus", tags=["Forecasting"])
def predict_model_consensus(student: StudentInput):
    """Run inference across all trained models in the tournament registry to quantify epistemic model agreement."""
    report = _read_benchmark_report()
    models_meta = report.get("models", [])
    df_row = pd.DataFrame([student.model_dump()])

    per_model = []
    scores: List[float] = []
    for m in models_meta:
        slug = m.get("slug")
        if not slug:
            continue
        try:
            pipeline, _ = get_pipeline(slug)
            raw_pred = float(pipeline.predict(df_row)[0])
            bounded = round(max(0.0, min(100.0, raw_pred)), 1)
            tier_name, _, _ = get_tier_details(bounded)
            scores.append(bounded)
            per_model.append({
                "slug": slug,
                "model_name": m.get("name", slug),
                "predicted_score": bounded,
                "grade_tier": tier_name,
            })
        except HTTPException:
            continue

    if not scores:
        pipeline, key = get_pipeline()
        bounded = round(float(np.clip(pipeline.predict(df_row)[0], 0.0, 100.0)), 1)
        tier_name, _, _ = get_tier_details(bounded)
        scores.append(bounded)
        per_model.append({
            "slug": key,
            "model_name": report.get("champion_model", "Champion Pipeline"),
            "predicted_score": bounded,
            "grade_tier": tier_name,
        })

    per_model.sort(key=lambda item: item["predicted_score"], reverse=True)
    mean_score = round(float(np.mean(scores)), 1)
    median_score = round(float(np.median(scores)), 1)
    std_dev = round(float(np.std(scores)), 2)
    min_score = round(float(np.min(scores)), 1)
    max_score = round(float(np.max(scores)), 1)

    if std_dev < 1.5:
        agreement = "HIGH"
    elif std_dev < 3.0:
        agreement = "MODERATE"
    else:
        agreement = "DIVERGENT"

    consensus_tier, consensus_label, icon = get_tier_details(mean_score)
    return {
        "models_evaluated": len(per_model),
        "consensus_mean_score": mean_score,
        "consensus_median_score": median_score,
        "score_std_dev": std_dev,
        "prediction_range": [min_score, max_score],
        "agreement_level": agreement,
        "grade_tier": consensus_tier,
        "tier_label": consensus_label,
        "status_icon": icon,
        "per_model_predictions": per_model,
    }


@app.post("/v1/predict/cohort-percentile", tags=["Forecasting"])
def predict_cohort_percentile(
    student: StudentInput,
    model_slug: Optional[str] = Query(default=None),
):
    """Benchmark a student's predicted exam score and study habits against the 20,000-student cohort distribution."""
    pipeline, _ = get_pipeline(model_slug)
    data_dict = student.model_dump()
    df_row = pd.DataFrame([data_dict])
    pred = float(pipeline.predict(df_row)[0])
    bounded_score = round(max(0.0, min(100.0, pred)), 1)

    report = _read_benchmark_report()
    ds_summary = report.get("dataset_summary", {})
    cohort_mean = float(ds_summary.get("target_mean", 73.5))
    cohort_std = max(float(ds_summary.get("target_std", 12.2)), 1.0)

    # Smooth logistic CDF approximation to normal percentile
    z_score = (bounded_score - cohort_mean) / cohort_std
    percentile = round(float(np.clip(100.0 / (1.0 + np.exp(-1.702 * z_score)), 0.5, 99.9)), 1)

    if percentile >= 90.0:
        standing = "Top 10% (Dean's Honors)"
    elif percentile >= 75.0:
        standing = "Top 25% (Upper Quartile)"
    elif percentile >= 50.0:
        standing = "Above Cohort Median"
    else:
        standing = "Below Cohort Median"

    tracked_keys = [
        "hours_studied",
        "attendance_percent",
        "previous_score",
        "quiz_average",
        "assignment_completion_rate",
        "sleep_hours",
    ]
    deltas = {
        k: round(float(data_dict[k]) - float(DEFAULT_STUDENT_PROFILE[k]), 2)
        for k in tracked_keys
        if k in data_dict and k in DEFAULT_STUDENT_PROFILE
    }

    tier_name, _, _ = get_tier_details(bounded_score)
    return {
        "predicted_score": bounded_score,
        "grade_tier": tier_name,
        "cohort_mean_score": round(cohort_mean, 2),
        "cohort_std_dev": round(cohort_std, 2),
        "z_score": round(z_score, 3),
        "estimated_percentile": percentile,
        "cohort_standing": standing,
        "feature_vs_cohort_deltas": deltas,
    }



@app.post("/v1/predict/cohort-benchmark", tags=["Analytics"])
def predict_cohort_benchmark(
    student: StudentInput,
    model_slug: Optional[str] = Query(default=None),
):
    """Compare a student's predicted performance and behavioral habits against the cohort median baseline."""
    pipeline, _ = get_pipeline(model_slug)
    student_dict = student.model_dump()
    baseline_dict = DEFAULT_STUDENT_PROFILE.copy()

    batch_df = pd.DataFrame([student_dict, baseline_dict])
    preds = np.clip(pipeline.predict(batch_df), 0.0, 100.0)
    student_score = round(float(preds[0]), 1)
    cohort_baseline_score = round(float(preds[1]), 1)
    score_delta = round(student_score - cohort_baseline_score, 1)

    # Normal-approximation percentile estimate relative to cohort distribution (std ~ 11.5 pts)
    z_score = (student_score - cohort_baseline_score) / 11.5
    estimated_percentile = round(float(np.clip(50.0 + 34.0 * np.tanh(0.85 * z_score), 1.0, 99.0)), 1)

    habit_deltas = {
        "hours_studied_delta": round(student.hours_studied - float(baseline_dict["hours_studied"]), 2),
        "attendance_percent_delta": round(student.attendance_percent - float(baseline_dict["attendance_percent"]), 2),
        "previous_score_delta": round(student.previous_score - float(baseline_dict["previous_score"]), 2),
        "sleep_hours_delta": round(student.sleep_hours - float(baseline_dict["sleep_hours"]), 2),
        "assignment_completion_delta": round(
            student.assignment_completion_rate - float(baseline_dict["assignment_completion_rate"]), 2
        ),
    }

    tier_name, _, _ = get_tier_details(student_score)
    return {
        "student_predicted_score": student_score,
        "cohort_baseline_score": cohort_baseline_score,
        "score_delta_vs_cohort": score_delta,
        "estimated_percentile": estimated_percentile,
        "standing_vs_cohort": "ABOVE_MEDIAN" if score_delta >= 0 else "BELOW_MEDIAN",
        "grade_tier": tier_name,
        "habit_deltas": habit_deltas,
    }


@app.post("/v1/predict/study-efficiency-audit", tags=["Analytics"])
def predict_study_efficiency_audit(
    student: StudentInput,
    model_slug: Optional[str] = Query(default=None),
):
    """Audit a student's cognitive study yield, sleep-recovery ratio, distraction drag, and burnout risk tier."""
    pipeline, _ = get_pipeline(model_slug)
    student_dict = student.model_dump()
    pred_score = round(float(np.clip(pipeline.predict(pd.DataFrame([student_dict]))[0], 0.0, 100.0)), 1)

    effective_study_hours = max(student.hours_studied, 0.5)
    score_per_study_hour = round(pred_score / effective_study_hours, 2)
    sleep_deficit_hours = round(max(0.0, 7.5 - student.sleep_hours), 2)
    sleep_recovery_ratio = round(student.sleep_hours / 8.0, 2)
    engagement_composite = round(
        0.5 * student.attendance_percent + 0.5 * student.assignment_completion_rate,
        1,
    )
    distraction_drag_pts = round(max(0.0, (student.screen_time_hours - 1.5) * 1.35), 2)

    if student.sleep_hours < 6.0 and (student.hours_studied >= 16.0 or student.stress_level >= 8):
        burnout_risk = "HIGH"
    elif student.sleep_hours < 6.8 or student.stress_level >= 7:
        burnout_risk = "MODERATE"
    else:
        burnout_risk = "LOW"

    tier_name, _, _ = get_tier_details(pred_score)
    return {
        "predicted_score": pred_score,
        "grade_tier": tier_name,
        "score_per_study_hour": score_per_study_hour,
        "sleep_deficit_hours": sleep_deficit_hours,
        "sleep_recovery_ratio": sleep_recovery_ratio,
        "engagement_composite": engagement_composite,
        "estimated_distraction_drag_pts": distraction_drag_pts,
        "burnout_risk_tier": burnout_risk,
    }


@app.post("/v1/predict/grade-tier-readiness", tags=["Analytics"])
def predict_grade_tier_readiness(
    student: StudentInput,
    model_slug: Optional[str] = Query(default=None),
):
    """Quantify a student's proximity to the next higher academic grade tier and current tier safety buffer."""
    pipeline, _ = get_pipeline(model_slug)
    student_dict = student.model_dump()
    pred_score = round(float(np.clip(pipeline.predict(pd.DataFrame([student_dict]))[0], 0.0, 100.0)), 1)

    tier_thresholds = [
        (90.0, "A+ (Distinction)"),
        (80.0, "A (First-Class Honors)"),
        (70.0, "B (Proficient)"),
        (60.0, "C (Satisfactory)"),
        (0.0, "D/F (At-Risk Intervention)"),
    ]

    current_floor = 0.0
    next_tier_name = "Maximum Tier Achieved"
    next_tier_threshold = 100.0
    for idx, (floor, label) in enumerate(tier_thresholds):
        if pred_score >= floor:
            current_floor = floor
            if idx > 0:
                next_tier_threshold, next_tier_name = tier_thresholds[idx - 1]
            break

    points_to_next_tier = round(max(0.0, next_tier_threshold - pred_score), 1) if pred_score < 90.0 else 0.0
    buffer_above_floor = round(max(0.0, pred_score - current_floor), 1)
    # Approximate marginal gain ~ 1.45 score pts per additional study hour
    est_extra_study_hours = round(points_to_next_tier / 1.45, 1) if points_to_next_tier > 0 else 0.0

    tier_name, tier_label, _ = get_tier_details(pred_score)
    return {
        "predicted_score": pred_score,
        "current_grade_tier": tier_name,
        "current_tier_label": tier_label,
        "current_tier_floor": current_floor,
        "buffer_above_tier_floor": buffer_above_floor,
        "next_grade_tier": next_tier_name,
        "next_tier_threshold": next_tier_threshold,
        "points_to_next_tier": points_to_next_tier,
        "estimated_additional_study_hours": est_extra_study_hours,
        "at_top_tier": pred_score >= 90.0,
    }


@app.post("/v1/predict/attendance-coursework-synergy", tags=["Analytics"])
def predict_attendance_coursework_synergy(
    student: StudentInput,
    model_slug: Optional[str] = Query(default=None),
):
    """Quantify the harmonic synergy between lecture attendance and assignment completion and project 95%+ mastery gains."""
    pipeline, _ = get_pipeline(model_slug)
    student_dict = student.model_dump()
    pred_score = round(float(np.clip(pipeline.predict(pd.DataFrame([student_dict]))[0], 0.0, 100.0)), 1)

    att = max(float(student.attendance_percent), 1.0)
    comp = max(float(student.assignment_completion_rate), 1.0)
    harmonic_synergy = round((2.0 * att * comp) / (att + comp), 2)
    formative_momentum = round(float(student.quiz_average) - float(student.previous_score), 1)

    optimized_dict = student_dict.copy()
    optimized_dict["attendance_percent"] = max(att, 95.0)
    optimized_dict["assignment_completion_rate"] = max(comp, 95.0)
    projected_95 = round(float(np.clip(pipeline.predict(pd.DataFrame([optimized_dict]))[0], 0.0, 100.0)), 1)
    synergy_unlock_gain = round(max(0.0, projected_95 - pred_score), 1)

    if harmonic_synergy < 75.0:
        coverage_risk = "HIGH"
    elif harmonic_synergy < 86.0:
        coverage_risk = "MODERATE"
    else:
        coverage_risk = "LOW"

    tier_name, _, _ = get_tier_details(pred_score)
    return {
        "predicted_score": pred_score,
        "grade_tier": tier_name,
        "attendance_completion_synergy_index": harmonic_synergy,
        "formative_momentum_delta": formative_momentum,
        "syllabus_coverage_risk": coverage_risk,
        "projected_score_at_95_synergy": projected_95,
        "synergy_unlock_gain_pts": synergy_unlock_gain,
    }


@app.post("/v1/predict/sleep-cognitive-recovery", tags=["Analytics"])
def predict_sleep_cognitive_recovery(
    student: StudentInput,
    model_slug: Optional[str] = Query(default=None),
):
    """Quantify sleep-study cognitive equilibrium and project exam score gain under an optimal 8.0-hour sleep schedule with reduced screen fatigue."""
    pipeline, _ = get_pipeline(model_slug)
    student_dict = student.model_dump()
    pred_score = round(float(np.clip(pipeline.predict(pd.DataFrame([student_dict]))[0], 0.0, 100.0)), 1)

    sleep_hrs = float(student.sleep_hours)
    study_hrs = max(0.5, float(student.hours_studied))
    screen_hrs = float(student.screen_time_hours)

    sleep_deficit = round(max(0.0, 8.0 - sleep_hrs), 2)
    cognitive_retention_multiplier = round(
        float(np.clip(1.0 - 0.06 * sleep_deficit - 0.02 * max(0.0, screen_hrs - 2.0), 0.65, 1.05)),
        3,
    )
    sleep_to_study_ratio = round(sleep_hrs / study_hrs, 2)

    restored_dict = student_dict.copy()
    restored_dict["sleep_hours"] = 8.0
    restored_dict["screen_time_hours"] = min(screen_hrs, 2.0)
    restored_dict["stress_level"] = min(int(student.stress_level), 4)
    projected_restored = round(float(np.clip(pipeline.predict(pd.DataFrame([restored_dict]))[0], 0.0, 100.0)), 1)
    recovery_gain = round(max(0.0, projected_restored - pred_score), 1)

    if sleep_deficit >= 2.0 or screen_hrs >= 5.0:
        fatigue_tier = "HIGH_COGNITIVE_FATIGUE"
    elif sleep_deficit >= 1.0 or screen_hrs >= 3.5:
        fatigue_tier = "MODERATE_SLEEP_DEBT"
    else:
        fatigue_tier = "OPTIMAL_RECOVERY_BALANCE"

    tier_name, _, _ = get_tier_details(pred_score)
    return {
        "predicted_score": pred_score,
        "grade_tier": tier_name,
        "sleep_deficit_hours": sleep_deficit,
        "sleep_to_study_ratio": sleep_to_study_ratio,
        "cognitive_retention_multiplier": cognitive_retention_multiplier,
        "cognitive_fatigue_tier": fatigue_tier,
        "projected_score_with_optimal_recovery": projected_restored,
        "sleep_recovery_gain_pts": recovery_gain,
    }


@app.post("/v1/predict/practice-mastery-roi", tags=["Analytics"])
def predict_practice_mastery_roi(
    student: StudentInput,
    model_slug: Optional[str] = Query(default=None),
):
    """Quantify active-recall practice paper efficiency and tutoring ROI, projecting score gains at high-mastery practice volume."""
    pipeline, _ = get_pipeline(model_slug)
    student_dict = student.model_dump()
    pred_score = round(float(np.clip(pipeline.predict(pd.DataFrame([student_dict]))[0], 0.0, 100.0)), 1)

    papers = int(student.practice_papers_solved)
    tutoring = int(student.tutoring_sessions)
    study_hrs = max(0.5, float(student.hours_studied))

    active_recall_density = round((papers + 1.5 * tutoring) / study_hrs, 3)

    mastery_dict = student_dict.copy()
    mastery_dict["practice_papers_solved"] = max(papers, 10)
    mastery_dict["tutoring_sessions"] = max(tutoring, 3)
    mastery_dict["quiz_average"] = min(100.0, max(float(student.quiz_average), float(student.previous_score) + 6.0))
    projected_mastery = round(float(np.clip(pipeline.predict(pd.DataFrame([mastery_dict]))[0], 0.0, 100.0)), 1)
    mastery_gain_pts = round(max(0.0, projected_mastery - pred_score), 1)

    if papers >= 7 and active_recall_density >= 0.65:
        practice_tier = "HIGH_ACTIVE_RECALL_MASTERY"
    elif papers >= 4:
        practice_tier = "MODERATE_PRACTICE_EXPOSURE"
    else:
        practice_tier = "LOW_PRACTICE_VOLUME"

    tier_name, _, _ = get_tier_details(pred_score)
    return {
        "predicted_score": pred_score,
        "grade_tier": tier_name,
        "practice_papers_solved": papers,
        "tutoring_sessions": tutoring,
        "active_recall_density": active_recall_density,
        "practice_mastery_tier": practice_tier,
        "projected_score_at_full_practice": projected_mastery,
        "practice_mastery_gain_pts": mastery_gain_pts,
    }









# Mount Production React Frontend if built
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))
if os.path.exists(frontend_dist):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
