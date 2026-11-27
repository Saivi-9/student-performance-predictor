"""
Integration tests for Student Performance Predictor v2.0 FastAPI REST endpoints.
"""

from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert data["version"] == "2.0.0"
    assert data["models_available"] >= 9


def test_list_models_endpoint():
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert len(data["models"]) == 9
    assert data["champion_model"]


def test_predict_endpoint_full_and_model_switch():
    payload = {
        "hours_studied": 14.0,
        "attendance_percent": 92.0,
        "previous_score": 82.0,
        "quiz_average": 85.0,
        "assignment_completion_rate": 90.0,
        "practice_papers_solved": 6,
        "sleep_hours": 7.5,
        "tutoring_sessions": 2,
        "physical_activity_hours": 5.0,
        "screen_time_hours": 2.5,
        "stress_level": 3,
        "commute_minutes": 20,
        "parental_involvement": "High",
        "motivation_level": "High",
        "access_to_resources": "High",
        "parental_education": "Master+",
        "extracurricular": "Yes",
        "internet_access": "Yes",
        "school_type": "Private",
        "learning_style": "Visual",
    }
    res = client.post("/v1/predict?model_slug=lightgbm", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["predicted_score"] <= 100.0
    assert data["model_used"] == "LightGBM"
    assert len(data["confidence_interval"]) == 2
    assert len(data["engineered_features"]) == 8
    assert len(data["feature_contributions"]) > 0


def test_predict_batch_endpoint():
    payload = {
        "students": [
            {"hours_studied": 5.0, "previous_score": 55.0, "attendance_percent": 65.0},
            {"hours_studied": 16.0, "previous_score": 88.0, "attendance_percent": 95.0},
        ]
    }
    res = client.post("/v1/predict/batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][1]["predicted_score"] > data["predictions"][0]["predicted_score"]


def test_simulate_endpoint():
    payload = {
        "hours_studied": 5.0,
        "attendance_percent": 85.0,
        "previous_score": 75.0,
    }
    res = client.post(
        "/v1/simulate?feature_name=hours_studied&min_val=2.0&max_val=18.0&steps=5",
        json=payload,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["feature_name"] == "hours_studied"
    assert len(data["trajectory"]) == 5
    assert data["trajectory"][-1]["predicted_score"] >= data["trajectory"][0]["predicted_score"]


def test_benchmarks_and_dataset_and_catalog_endpoints():
    bench = client.get("/v1/benchmarks")
    assert bench.status_code == 200
    assert bench.json()["dataset_summary"]["total_rows"] >= 20000

    ds = client.get("/v1/dataset/summary?sample_limit=5")
    assert ds.status_code == 200
    assert len(ds.json()["sample_records"]) == 5

    cat = client.get("/v1/pretrained/catalog")
    assert cat.status_code == 200
    assert len(cat.json()["catalog"]) >= 5


def test_registry_diagnostics_endpoint():
    res = client.get("/v1/system/registry-diagnostics")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "READY"
    assert data["champion_artifact_available"] is True
    assert data["registry_count"] >= 9
    assert data["datasets"]["primary_20k"]["available"] is True


def test_intervention_plan_endpoint():
    payload = {
        "hours_studied": 6.0,
        "attendance_percent": 74.0,
        "previous_score": 68.0,
        "assignment_completion_rate": 72.0,
        "practice_papers_solved": 2,
        "tutoring_sessions": 0,
        "sleep_hours": 5.5,
    }
    res = client.post("/v1/predict/intervention-plan?target_score=80.0", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["combined_projected_score"] >= data["current_score"]
    assert len(data["ranked_interventions"]) == 6
    assert data["ranked_interventions"][0]["score_gain"] >= data["ranked_interventions"][-1]["score_gain"]


def test_model_consensus_endpoint():
    payload = {
        "hours_studied": 12.0,
        "attendance_percent": 90.0,
        "previous_score": 80.0,
        "quiz_average": 82.0,
        "assignment_completion_rate": 88.0,
        "practice_papers_solved": 5,
        "sleep_hours": 7.5,
    }
    res = client.post("/v1/predict/model-consensus", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["models_evaluated"] >= 9
    assert 0.0 <= data["consensus_mean_score"] <= 100.0
    assert data["agreement_level"] in ("HIGH", "MODERATE", "DIVERGENT")
    assert len(data["prediction_range"]) == 2
    assert data["prediction_range"][0] <= data["consensus_mean_score"] <= data["prediction_range"][1]
    assert len(data["per_model_predictions"]) == data["models_evaluated"]


def test_cohort_percentile_endpoint():
    payload = {
        "hours_studied": 16.0,
        "attendance_percent": 95.0,
        "previous_score": 88.0,
        "quiz_average": 90.0,
        "assignment_completion_rate": 94.0,
        "practice_papers_solved": 8,
        "sleep_hours": 8.0,
    }
    res = client.post("/v1/predict/cohort-percentile", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 0.5 <= data["estimated_percentile"] <= 99.9
    assert data["estimated_percentile"] > 50.0
    assert "cohort_standing" in data
    assert data["feature_vs_cohort_deltas"]["hours_studied"] > 0



def test_cohort_benchmark_endpoint():
    payload = {
        "hours_studied": 15.0,
        "attendance_percent": 94.0,
        "previous_score": 84.0,
        "quiz_average": 86.0,
        "assignment_completion_rate": 92.0,
        "practice_papers_solved": 6,
        "sleep_hours": 7.8,
    }
    res = client.post("/v1/predict/cohort-benchmark", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["student_predicted_score"] <= 100.0
    assert 1.0 <= data["estimated_percentile"] <= 99.0
    assert data["standing_vs_cohort"] in ("ABOVE_MEDIAN", "BELOW_MEDIAN")
    assert "hours_studied_delta" in data["habit_deltas"]


def test_study_efficiency_audit_endpoint():
    payload = {
        "hours_studied": 18.0,
        "attendance_percent": 92.0,
        "previous_score": 81.0,
        "quiz_average": 84.0,
        "assignment_completion_rate": 90.0,
        "practice_papers_solved": 6,
        "sleep_hours": 5.5,
        "screen_time_hours": 3.5,
        "stress_level": 8,
    }
    res = client.post("/v1/predict/study-efficiency-audit", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["predicted_score"] <= 100.0
    assert data["score_per_study_hour"] > 0.0
    assert data["sleep_deficit_hours"] == 2.0
    assert data["engagement_composite"] == 91.0
    assert data["estimated_distraction_drag_pts"] == 2.7
    assert data["burnout_risk_tier"] == "HIGH"


def test_grade_tier_readiness_endpoint():
    payload = {
        "hours_studied": 11.0,
        "attendance_percent": 86.0,
        "previous_score": 76.0,
        "quiz_average": 78.0,
        "assignment_completion_rate": 84.0,
        "practice_papers_solved": 4,
        "sleep_hours": 7.5,
    }
    res = client.post("/v1/predict/grade-tier-readiness", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["predicted_score"] <= 100.0
    assert data["buffer_above_tier_floor"] >= 0.0
    assert data["points_to_next_tier"] >= 0.0
    assert data["estimated_additional_study_hours"] >= 0.0
    assert isinstance(data["at_top_tier"], bool)


def test_attendance_coursework_synergy_endpoint():
    payload = {
        "hours_studied": 10.0,
        "attendance_percent": 72.0,
        "previous_score": 74.0,
        "quiz_average": 78.5,
        "assignment_completion_rate": 70.0,
        "practice_papers_solved": 4,
        "sleep_hours": 7.5,
    }
    res = client.post("/v1/predict/attendance-coursework-synergy", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["predicted_score"] <= 100.0
    assert data["attendance_completion_synergy_index"] == 70.99
    assert data["formative_momentum_delta"] == 4.5
    assert data["syllabus_coverage_risk"] == "HIGH"
    assert data["projected_score_at_95_synergy"] >= data["predicted_score"]
    assert data["synergy_unlock_gain_pts"] >= 0.0


def test_sleep_cognitive_recovery_endpoint():
    payload = {
        "hours_studied": 12.0,
        "attendance_percent": 88.0,
        "previous_score": 76.0,
        "quiz_average": 78.0,
        "assignment_completion_rate": 85.0,
        "practice_papers_solved": 4,
        "sleep_hours": 5.5,
        "screen_time_hours": 5.5,
        "stress_level": 8,
    }
    res = client.post("/v1/predict/sleep-cognitive-recovery", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["predicted_score"] <= 100.0
    assert data["sleep_deficit_hours"] == 2.5
    assert data["cognitive_fatigue_tier"] == "HIGH_COGNITIVE_FATIGUE"
    assert 0.65 <= data["cognitive_retention_multiplier"] <= 1.0
    assert data["projected_score_with_optimal_recovery"] >= data["predicted_score"]
    assert data["sleep_recovery_gain_pts"] >= 0.0


def test_practice_mastery_roi_endpoint():
    payload = {
        "hours_studied": 10.0,
        "attendance_percent": 85.0,
        "previous_score": 72.0,
        "quiz_average": 73.0,
        "assignment_completion_rate": 82.0,
        "practice_papers_solved": 2,
        "tutoring_sessions": 0,
        "sleep_hours": 7.5,
    }
    res = client.post("/v1/predict/practice-mastery-roi", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["predicted_score"] <= 100.0
    assert data["practice_mastery_tier"] == "LOW_PRACTICE_VOLUME"
    assert data["projected_score_at_full_practice"] >= data["predicted_score"]
    assert data["practice_mastery_gain_pts"] >= 0.0









