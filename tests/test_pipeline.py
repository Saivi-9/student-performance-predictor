"""
Unit tests for data validation, preprocessing pipeline, and model inference.
"""

import pytest
import pandas as pd
import numpy as np
from data.generate_dataset import generate_student_data
from src.dataset import validate_data, split_data
from src.pipeline import build_preprocessor, create_model_pipeline
from src.models import get_candidate_models
from predict import get_performance_tier, predict_score


@pytest.fixture
def sample_data():
    """Generate reproducible sample dataset for testing."""
    return generate_student_data(n_samples=100, random_seed=99)


def test_generate_data_shape(sample_data):
    """Verify dataset dimensions and expected columns."""
    assert len(sample_data) == 100
    assert 'final_score' in sample_data.columns
    assert 'hours_studied' in sample_data.columns
    assert 'attendance_percent' in sample_data.columns


def test_validate_data_detects_columns(sample_data):
    """Verify data validation detects missing values and columns properly."""
    summary = validate_data(sample_data)
    assert summary['total_rows'] == 100
    assert summary['target_column'] == 'final_score'
    assert 'hours_studied' in summary['numerical_columns']


def test_split_data_no_leakage(sample_data):
    """Verify split proportions and absence of index overlap."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        sample_data, test_size=0.2, val_size=0.1, random_state=42
    )

    # Check non-empty
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    # Ensure total sums to total clean samples
    total_split = len(X_train) + len(X_val) + len(X_test)
    assert total_split == len(sample_data.dropna(subset=['final_score']))

    # Ensure no index leakage
    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)
    assert len(train_idx.intersection(val_idx)) == 0
    assert len(train_idx.intersection(test_idx)) == 0
    assert len(val_idx.intersection(test_idx)) == 0


def test_pipeline_preprocessor(sample_data):
    """Verify preprocessing imputes and encodes properly without NaNs."""
    X_train, _, _, y_train, _, _ = split_data(sample_data, test_size=0.2, val_size=0.1)
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X_train)
    # Check transformed array is 2D numpy array and contains no NaNs
    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(transformed).any()


def test_candidate_models_train_and_predict(sample_data):
    """Verify all candidate models fit and output valid numerical predictions."""
    X_train, X_val, _, y_train, y_val, _ = split_data(sample_data, test_size=0.2, val_size=0.1)
    models = get_candidate_models(random_state=42)

    for name, model in models.items():
        pipeline = create_model_pipeline(model)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)

        assert len(preds) == len(X_val)
        assert np.isfinite(preds).all()


def test_performance_tier():
    """Verify grade tier mappings."""
    assert "Grade A" in get_performance_tier(92.5)
    assert "Grade B" in get_performance_tier(84.0)
    assert "Grade C" in get_performance_tier(71.0)
    assert "Grade D" in get_performance_tier(63.0)
    assert "Grade F" in get_performance_tier(45.0)
