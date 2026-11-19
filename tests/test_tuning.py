"""
Unit tests for automated hyperparameter tuning module (Ridge, RF, LightGBM).
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from data.generate_dataset import generate_student_data
from src.dataset import split_data
from src.tuning import tune_ridge_pipeline, tune_random_forest_pipeline, tune_lightgbm_pipeline


def _make_dummy_dataset():
    df = generate_student_data(n_samples=120, random_seed=42)
    X_train, _, _, y_train, _, _ = split_data(df, test_size=0.15, val_size=0.15, random_state=42)
    return X_train, y_train


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
    preds = best_pipeline.predict(X.head(5))
    assert len(preds) == 5


def test_tune_random_forest_pipeline_fast():
    X, y = _make_dummy_dataset()
    param_grid = {
        "model__n_estimators": [10, 20],
        "model__max_depth": [3, 5],
    }

    best_pipeline, best_params, results_df = tune_random_forest_pipeline(
        X, y, param_grid=param_grid, cv=2, random_state=42
    )

    assert isinstance(best_pipeline, Pipeline)
    assert "model__n_estimators" in best_params
    assert len(results_df) == 4
    preds = best_pipeline.predict(X.head(5))
    assert len(preds) == 5


def test_tune_lightgbm_pipeline_fast():
    X, y = _make_dummy_dataset()
    param_distributions = {
        "model__n_estimators": [15, 25],
        "model__learning_rate": [0.05, 0.1],
    }

    best_pipeline, best_params, results_df = tune_lightgbm_pipeline(
        X, y, param_distributions=param_distributions, n_iter=2, cv=2, random_state=42
    )

    assert isinstance(best_pipeline, Pipeline)
    assert "model__n_estimators" in best_params
    assert len(results_df) == 2
    preds = best_pipeline.predict(X.head(5))
    assert len(preds) == 5
