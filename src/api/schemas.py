"""
Pydantic Schemas for Student Academic Performance Prediction API (v2.0).
Supports all 20 raw features with backward-compatible defaults, model selection,
batch prediction, explainability breakdowns, and Hugging Face Hub integration.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    hours_studied: float = Field(12.0, ge=0.0, le=40.0, json_schema_extra={"example": 12.0})
    attendance_percent: float = Field(88.0, ge=0.0, le=100.0, json_schema_extra={"example": 88.0})
    previous_score: float = Field(74.0, ge=0.0, le=100.0, json_schema_extra={"example": 74.0})
    quiz_average: float = Field(78.0, ge=0.0, le=100.0, json_schema_extra={"example": 78.0})
    assignment_completion_rate: float = Field(85.0, ge=0.0, le=100.0, json_schema_extra={"example": 85.0})
    practice_papers_solved: float = Field(5.0, ge=0.0, le=20.0, json_schema_extra={"example": 5.0})
    sleep_hours: float = Field(7.5, ge=2.0, le=16.0, json_schema_extra={"example": 7.5})
    tutoring_sessions: int = Field(2, ge=0, le=20, json_schema_extra={"example": 2})
    physical_activity_hours: float = Field(4.5, ge=0.0, le=25.0, json_schema_extra={"example": 4.5})
    screen_time_hours: float = Field(3.0, ge=0.0, le=16.0, json_schema_extra={"example": 3.0})
    stress_level: float = Field(4.0, ge=1.0, le=10.0, json_schema_extra={"example": 4.0})
    commute_minutes: float = Field(25.0, ge=0.0, le=180.0, json_schema_extra={"example": 25.0})
    parental_involvement: str = Field("Medium", json_schema_extra={"example": "Medium"})
    motivation_level: str = Field("Medium", json_schema_extra={"example": "Medium"})
    access_to_resources: str = Field("Medium", json_schema_extra={"example": "Medium"})
    parental_education: str = Field("Bachelor", json_schema_extra={"example": "Bachelor"})
    extracurricular: str = Field("Yes", json_schema_extra={"example": "Yes"})
    internet_access: str = Field("Yes", json_schema_extra={"example": "Yes"})
    school_type: str = Field("Public", json_schema_extra={"example": "Public"})
    learning_style: str = Field("Visual", json_schema_extra={"example": "Visual"})


class FactorContribution(BaseModel):
    feature: str
    label: str
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    predicted_score: float
    confidence_interval: List[float]
    model_used: str
    grade_tier: str
    tier_label: str
    status_icon: str
    delta_vs_previous: float
    effort_density_hrs_per_session: float
    engineered_features: Dict[str, float]
    feature_contributions: List[FactorContribution]
    behavioral_diagnostics: Dict[str, Any]
    input_features: Dict[str, Any]


class BatchPredictionRequest(BaseModel):
    students: List[StudentInput]
    model_slug: Optional[str] = None


class BatchPredictionItem(BaseModel):
    index: int
    predicted_score: float
    grade_tier: str
    academic_momentum: float
    engagement_score: float


class BatchPredictionResponse(BaseModel):
    count: int
    model_used: str
    average_score: float
    at_risk_count: int
    predictions: List[BatchPredictionItem]


class SimulationStep(BaseModel):
    value: float
    predicted_score: float


class SimulationResponse(BaseModel):
    feature_name: str
    model_used: str
    trajectory: List[SimulationStep]
    current_value: float
    current_score: float
