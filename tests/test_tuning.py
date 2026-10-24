"""
Unit tests for automated hyperparameter tuning module.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.tuning import tune_ridge_pipeline, tune_random_forest_pipeline


def _make_dummy_dataset():
    np.random.seed(42)
    n = 80
    df = pd.DataFrame({
        'hours_studied': np.random.uniform(2, 20, size=n),
        'attendance_percent': np.random.uniform(60, 100, size=n),
        'previous_score': np.random.uniform(40, 95, size=n),
        'sleep_hours': np.random.uniform(5, 9, size=n),
        'tutoring_sessions': np.random.randint(0, 5, size=n),
        'parental_involvement': np.random.choice(['Low', 'Medium', 'High'], size=n),
        'extracurricular': np.random.choice(['Yes', 'No'], size=n),
        'internet_access': np.random.choice(['Yes', 'No'], size=n),
    })
    y = (
        0.5 * df['previous_score'] +
        1.2 * df['hours_studied'] +
        np.random.normal(0, 2, size=n)
    )
    return df, y


def test_tune_ridge_pipeline():
    X, y = _make_dummy_dataset()
    alphas = [0.1, 1.0, 10.0]

    best_pipeline, best_params, results_df = tune_ridge_pipeline(
        X, y, alphas=alphas, cv=3, random_state=42
    )

    assert isinstance(best_pipeline, Pipeline)
    assert "model__alpha" in best_params
    assert best_params["model__alpha"] in alphas
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == len(alphas)
    assert "mean_test_rmse" in results_df.columns
    # Check that predictions work on new inputs
    preds = best_pipeline.predict(X.head(5))
    assert len(preds) == 5


def test_tune_random_forest_pipeline_fast():
    X, y = _make_dummy_dataset()
    param_grid = {
        "model__n_estimators": [10, 20],
        "model__max_depth": [3, 5]
    }

    best_pipeline, best_params, results_df = tune_random_forest_pipeline(
        X, y, param_grid=param_grid, cv=2, random_state=42
    )

    assert isinstance(best_pipeline, Pipeline)
    assert "model__n_estimators" in best_params
    assert len(results_df) == 4
    preds = best_pipeline.predict(X.head(5))
    assert len(preds) == 5
