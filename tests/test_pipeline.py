"""
Unit tests for data validation, preprocessing pipeline, 9 model architectures, and inference.
"""

import pytest
import numpy as np
from data.generate_dataset import generate_student_data
from src.dataset import validate_data, split_data, EXPECTED_NUMERICAL_COLS, EXPECTED_CATEGORICAL_COLS
from src.pipeline import build_preprocessor, create_model_pipeline
from src.models import get_candidate_models
from predict import get_performance_tier


@pytest.fixture
def sample_data():
    """Generate reproducible 20-feature sample dataset for testing."""
    return generate_student_data(n_samples=120, random_seed=99)


def test_generate_data_shape(sample_data):
    """Verify dataset dimensions and expected 20 raw feature columns + target."""
    assert len(sample_data) == 120
    assert 'final_score' in sample_data.columns
    for col in EXPECTED_NUMERICAL_COLS + EXPECTED_CATEGORICAL_COLS:
        assert col in sample_data.columns


def test_validate_data_detects_columns(sample_data):
    """Verify data validation detects missing values and 20 raw columns properly."""
    summary = validate_data(sample_data)
    assert summary['total_rows'] == 120
    assert summary['target_column'] == 'final_score'
    assert len(summary['numerical_columns']) == 12
    assert len(summary['categorical_columns']) == 8


def test_split_data_no_leakage(sample_data):
    """Verify split proportions and absence of index overlap."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        sample_data, test_size=0.2, val_size=0.1, random_state=42
    )

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    total_split = len(X_train) + len(X_val) + len(X_test)
    assert total_split == len(sample_data.dropna(subset=['final_score']))

    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)
    assert len(train_idx.intersection(val_idx)) == 0
    assert len(train_idx.intersection(test_idx)) == 0
    assert len(val_idx.intersection(test_idx)) == 0


def test_pipeline_preprocessor(sample_data):
    """Verify preprocessing + feature engineering outputs clean 2D float matrix with no NaNs."""
    from src.features import AcademicFeatureEngineer

    X_train, _, _, _, _, _ = split_data(sample_data, test_size=0.2, val_size=0.1)
    eng = AcademicFeatureEngineer()
    preprocessor = build_preprocessor()

    X_eng = eng.fit_transform(X_train)
    transformed = preprocessor.fit_transform(X_eng)
    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(transformed).any()


def test_candidate_models_train_and_predict(sample_data):
    """Verify all 9 candidate architectures exist, fit, and output finite predictions."""
    X_train, X_val, _, y_train, _, _ = split_data(sample_data, test_size=0.2, val_size=0.1)
    models = get_candidate_models(random_state=42)
    assert len(models) == 9
    assert "LightGBM" in models
    assert "Stacking Ensemble" in models

    for name, model in models.items():
        pipeline = create_model_pipeline(model)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)

        assert len(preds) == len(X_val), f"{name} returned wrong prediction count"
        assert np.isfinite(preds).all(), f"{name} produced non-finite predictions"


def test_performance_tier():
    """Verify grade tier mappings."""
    assert "Grade A" in get_performance_tier(92.5)
    assert "Grade B" in get_performance_tier(84.0)
    assert "Grade C" in get_performance_tier(71.0)
    assert "Grade D" in get_performance_tier(63.0)
    assert "Grade F" in get_performance_tier(45.0)
