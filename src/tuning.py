"""
Automated Hyperparameter Optimization Module
---------------------------------------------
Performs systematic hyperparameter grid tuning with cross-validation
for Ridge, Random Forest, and LightGBM regression pipelines.
"""

from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from src.pipeline import create_model_pipeline


def tune_ridge_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alphas: list = None,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """Tune L2 regularization alpha for Ridge regression pipeline."""
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
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    results_df = pd.DataFrame({
        "alpha": grid_search.cv_results_["param_model__alpha"].data,
        "mean_test_rmse": -grid_search.cv_results_["mean_test_score"],
        "std_test_rmse": grid_search.cv_results_["std_test_score"],
        "mean_train_rmse": -grid_search.cv_results_["mean_train_score"],
    }).sort_values(by="mean_test_rmse").reset_index(drop=True)

    return grid_search.best_estimator_, grid_search.best_params_, results_df


def tune_random_forest_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = None,
    cv: int = 3,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """Perform grid search optimization on Random Forest hyperparameters."""
    if param_grid is None:
        param_grid = {
            "model__n_estimators": [80, 120, 160],
            "model__max_depth": [8, 12, 16],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        }

    base_pipeline = create_model_pipeline(RandomForestRegressor(random_state=random_state, n_jobs=-1))
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=kf,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    cv_results = grid_search.cv_results_
    results_df = pd.DataFrame({
        "params": [str(p) for p in cv_results["params"]],
        "mean_test_rmse": -cv_results["mean_test_score"],
        "std_test_rmse": cv_results["std_test_score"],
    }).sort_values(by="mean_test_rmse").reset_index(drop=True)

    return grid_search.best_estimator_, grid_search.best_params_, results_df


def tune_lightgbm_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_distributions: dict = None,
    n_iter: int = 12,
    cv: int = 3,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """Perform randomized search optimization on LightGBM hyperparameters."""
    if param_distributions is None:
        param_distributions = {
            "model__n_estimators": [150, 250, 350],
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__num_leaves": [20, 31, 45],
            "model__subsample": [0.75, 0.85, 1.0],
            "model__colsample_bytree": [0.75, 0.85, 1.0],
        }

    base_pipeline = create_model_pipeline(
        LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1)
    )
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=kf,
        n_jobs=-1,
        random_state=random_state,
    )
    search.fit(X_train, y_train)

    cv_results = search.cv_results_
    results_df = pd.DataFrame({
        "params": [str(p) for p in cv_results["params"]],
        "mean_test_rmse": -cv_results["mean_test_score"],
        "std_test_rmse": cv_results["std_test_score"],
    }).sort_values(by="mean_test_rmse").reset_index(drop=True)

    return search.best_estimator_, search.best_params_, results_df
