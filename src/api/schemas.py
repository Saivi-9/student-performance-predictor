"""
Pydantic Schemas for Student Academic Performance Prediction API.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    hours_studied: float = Field(..., ge=0.0, le=40.0, json_schema_extra={"example": 6.5})
    attendance_percent: float = Field(..., ge=0.0, le=100.0, json_schema_extra={"example": 88.0})
    previous_score: float = Field(..., ge=0.0, le=100.0, json_schema_extra={"example": 76.0})
    sleep_hours: float = Field(7.0, ge=2.0, le=16.0, json_schema_extra={"example": 7.5})
    tutoring_sessions: int = Field(2, ge=0, le=20, json_schema_extra={"example": 2})
    parental_involvement: str = Field("Medium", json_schema_extra={"example": "Medium"})
    extracurricular: str = Field("Yes", json_schema_extra={"example": "Yes"})
    internet_access: str = Field("Yes", json_schema_extra={"example": "Yes"})


class PredictionResponse(BaseModel):
    predicted_score: float
    grade_tier: str
    tier_label: str
    status_icon: str
    delta_vs_previous: float
    effort_density_hrs_per_session: float
    behavioral_diagnostics: Dict[str, Any]
    input_features: Dict[str, Any]


class SimulationStep(BaseModel):
    value: float
    predicted_score: float


class SimulationResponse(BaseModel):
    feature_name: str
    trajectory: List[SimulationStep]
    current_value: float
    current_score: float
