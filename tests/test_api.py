"""
Unit tests for Student Performance Predictor FastAPI REST endpoints.
"""

from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert "version" in data


def test_predict_endpoint():
    payload = {
        "hours_studied": 8.0,
        "attendance_percent": 90.0,
        "previous_score": 82.0,
        "sleep_hours": 7.5,
        "tutoring_sessions": 2,
        "parental_involvement": "High",
        "extracurricular": "Yes",
        "internet_access": "Yes"
    }
    res = client.post("/v1/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_score" in data
    assert 0.0 <= data["predicted_score"] <= 100.0
    assert "grade_tier" in data
    assert "effort_density_hrs_per_session" in data
    assert "behavioral_diagnostics" in data


def test_simulate_endpoint():
    payload = {
        "hours_studied": 5.0,
        "attendance_percent": 85.0,
        "previous_score": 75.0,
        "sleep_hours": 7.0,
        "tutoring_sessions": 1,
        "parental_involvement": "Medium",
        "extracurricular": "Yes",
        "internet_access": "Yes"
    }
    res = client.post(
        "/v1/simulate?feature_name=hours_studied&min_val=2.0&max_val=16.0&steps=5",
        json=payload
    )
    assert res.status_code == 200
    data = res.json()
    assert data["feature_name"] == "hours_studied"
    assert len(data["trajectory"]) == 5
    assert data["trajectory"][-1]["predicted_score"] >= data["trajectory"][0]["predicted_score"]
