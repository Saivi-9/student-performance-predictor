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
