"""
Unit tests for Hugging Face Pre-Trained Model Hub integration utilities.
"""

import pandas as pd
import numpy as np
from src.pretrained_hub import (
    PRETRAINED_CATALOG,
    _scale_for_shalan1,
    _prepare_uci_features_for_sergey,
)


def test_pretrained_catalog_structure():
    assert len(PRETRAINED_CATALOG) >= 5
    repo_ids = {item["repo_id"] for item in PRETRAINED_CATALOG}
    assert "Shalan1/student-performance-predictor-models" in repo_ids
    assert "SergeyR256/student-performance" in repo_ids


def test_scale_for_shalan1():
    df = pd.DataFrame([{
        "Hours Studied": 5.0,
        "Previous Scores": 70.0,
        "Extracurricular Activities": 1.0,
        "Sleep Hours": 7.0,
        "Sample Question Papers Practiced": 5.0,
    }])
    scaled = _scale_for_shalan1(df)
    assert list(scaled.columns) == [
        "Hours Studied",
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced",
    ]
    assert np.isfinite(scaled.values).all()


def test_prepare_uci_features_for_sergey():
    df = pd.DataFrame([{
        "Dalc": 1,
        "Walc": 2,
        "Medu": 4,
        "Fedu": 3,
        "goout": 2,
        "studytime": 3,
        "failures": 1,
        "age": 17,
        "health": 4,
        "freetime": 3,
    }])
    feat = _prepare_uci_features_for_sergey(df)
    for col in [
        "total_alcohol",
        "parent_edu_avg",
        "goout_studytime_ratio",
        "alcohol_study_interaction",
        "failures_squared",
        "age_squared",
        "health_freetime_ratio",
    ]:
        assert col in feat.columns
    assert feat["total_alcohol"].iloc[0] == 3
    assert feat["parent_edu_avg"].iloc[0] == 3.5
