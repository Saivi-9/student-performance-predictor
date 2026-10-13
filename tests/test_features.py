"""Unit tests for Feature Engineering module."""

import pytest
import pandas as pd
import numpy as np
from data.generate_dataset import generate_student_data
from src.features import engineer_features, ENGINEERED_FEATURES


@pytest.fixture
def sample_data():
    return generate_student_data(n_samples=100, random_seed=42)


def test_engineer_features_adds_columns(sample_data):
    df_eng = engineer_features(sample_data)
    for col in ENGINEERED_FEATURES:
        assert col in df_eng.columns, f"Missing engineered feature: {col}"


def test_engineer_features_no_nans(sample_data):
    df_eng = engineer_features(sample_data)
    for col in ENGINEERED_FEATURES:
        assert not df_eng[col].isnull().any(), f"NaNs found in engineered feature: {col}"


def test_engineer_features_does_not_mutate_original(sample_data):
    original_cols = list(sample_data.columns)
    _ = engineer_features(sample_data)
    assert list(sample_data.columns) == original_cols


def test_study_efficiency_non_negative(sample_data):
    df_eng = engineer_features(sample_data)
    assert (df_eng['study_efficiency'] >= 0).all()


def test_engagement_score_bounded(sample_data):
    df_eng = engineer_features(sample_data)
    assert (df_eng['engagement_score'] >= 0).all()
    assert (df_eng['engagement_score'] <= 1.0).all()


def test_sleep_deficit_non_negative(sample_data):
    df_eng = engineer_features(sample_data)
    assert (df_eng['sleep_deficit'] >= 0).all()
