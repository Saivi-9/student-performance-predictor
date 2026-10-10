"""
Models module defining candidate machine learning regression models and cross-validation routines.
"""

from typing import Dict
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import BaseEstimator
from src.pipeline import create_model_pipeline


def get_candidate_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Define a collection of diverse regression models for benchmarking.

    1. Linear Regression: Baseline interpretable linear model.
    2. Ridge Regression: Regularized linear model (L2 penalty) to prevent overfitting.
    3. Random Forest Regressor: Ensemble tree-based model capable of capturing non-linear interactions.

    Args:
        random_state: Seed for reproducible models.

    Returns:
        Dict[str, BaseEstimator]: Dictionary of model names to model estimators.
    """
    return {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=random_state),
        'Random Forest': RandomForestRegressor(
            n_estimators=120,
            max_depth=7,
            min_samples_split=4,
            random_state=random_state
        )
    }


def perform_cross_validation(
    model: BaseEstimator,
    X_train,
    y_train,
    cv: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """Perform k-fold cross-validation on training data using an end-to-end pipeline.

    Calculates Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.

    Args:
        model: Base estimator to benchmark.
        X_train: Training features DataFrame.
        y_train: Training target Series.
        cv: Number of folds (default: 5).
        random_state: Random state seed.

    Returns:
        Dict[str, float]: Cross-validation score averages and standard deviations.
    """
    pipeline = create_model_pipeline(model)
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Scikit-learn scoring metrics for regression
    neg_mae_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
    neg_rmse_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
    r2_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='r2')

    return {
        'cv_mae_mean': float(-np.mean(neg_mae_scores)),
        'cv_mae_std': float(np.std(neg_mae_scores)),
        'cv_rmse_mean': float(-np.mean(neg_rmse_scores)),
        'cv_rmse_std': float(np.std(neg_rmse_scores)),
        'cv_r2_mean': float(np.mean(r2_scores)),
        'cv_r2_std': float(np.std(r2_scores)),
    }
