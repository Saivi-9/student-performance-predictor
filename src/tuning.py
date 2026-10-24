"""
Automated Hyperparameter Optimization Module
---------------------------------------------
Performs systematic hyperparameter grid tuning with cross-validation
for regression pipelines to extract optimal predictive performance.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from src.pipeline import create_model_pipeline


def tune_ridge_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alphas: list = None,
    cv: int = 5,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """
    Tunes L2 regularization parameter (alpha) for Ridge regression pipeline.

    Args:
        X_train: Training features.
        y_train: Target values.
        alphas: List of candidate alpha values to evaluate.
        cv: Number of cross-validation folds.
        random_state: Seed for fold shuffling.

    Returns:
        Tuple:
            - Best fitted Pipeline
            - Best parameters dict
            - Summary DataFrame of candidate parameters and scores
    """
    if alphas is None:
        alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]

    base_pipeline = create_model_pipeline(Ridge(random_state=random_state))
    param_grid = {"model__alpha": alphas}

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=kf,
        n_jobs=-1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    results_df = pd.DataFrame({
        "alpha": grid_search.cv_results_["param_model__alpha"].data,
        "mean_test_rmse": -grid_search.cv_results_["mean_test_score"],
        "std_test_rmse": grid_search.cv_results_["std_test_score"],
        "mean_train_rmse": -grid_search.cv_results_["mean_train_score"]
    }).sort_values(by="mean_test_rmse").reset_index(drop=True)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_pipeline, best_params, results_df


def tune_random_forest_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = None,
    cv: int = 3,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """
    Performs grid search optimization on Random Forest hyperparameters.

    Args:
        X_train: Training features.
        y_train: Target values.
        param_grid: Dictionary mapping parameter names to value grids.
        cv: Number of cross-validation folds.
        random_state: Seed for fold shuffling.

    Returns:
        Tuple:
            - Best fitted Pipeline
            - Best parameters dict
            - Summary DataFrame of candidate parameters and scores
    """
    if param_grid is None:
        param_grid = {
            "model__n_estimators": [60, 100, 150],
            "model__max_depth": [5, 8, 12, None],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        }

    base_pipeline = create_model_pipeline(RandomForestRegressor(random_state=random_state))
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=kf,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    cv_results = grid_search.cv_results_
    results_df = pd.DataFrame({
        "params": [str(p) for p in cv_results["params"]],
        "mean_test_rmse": -cv_results["mean_test_score"],
        "std_test_rmse": cv_results["std_test_score"]
    }).sort_values(by="mean_test_rmse").reset_index(drop=True)

    return grid_search.best_estimator_, grid_search.best_params_, results_df
