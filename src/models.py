"""
Models module defining 9 diverse regression architectures, stacking ensemble,
and k-fold cross-validation benchmarking routines.
"""

from typing import Dict
import re
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import BaseEstimator
from lightgbm import LGBMRegressor

from src.pipeline import create_model_pipeline


def model_slug(name: str) -> str:
    """Convert display model name to filesystem/URL-safe slug."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def get_candidate_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Define 9 diverse machine learning models spanning linear, regularized,
    tree ensembles, gradient boosting (LightGBM + HistGBDT), neural network, and stacking.
    """
    return {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=random_state),
        'ElasticNet': ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=random_state),
        'Random Forest': RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_split=4,
            n_jobs=-1,
            random_state=random_state,
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_split=4,
            n_jobs=-1,
            random_state=random_state,
        ),
        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.06,
            max_depth=6,
            random_state=random_state,
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
        'MLP Neural Network': MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            alpha=0.01,
            max_iter=250,
            early_stopping=True,
            random_state=random_state,
        ),
        'Stacking Ensemble': StackingRegressor(
            estimators=[
                ('ridge', Ridge(alpha=1.0, random_state=random_state)),
                ('hist_gb', HistGradientBoostingRegressor(
                    max_iter=180, learning_rate=0.06, max_depth=6, random_state=random_state
                )),
                ('lgbm', LGBMRegressor(
                    n_estimators=200, learning_rate=0.05, num_leaves=31,
                    random_state=random_state, n_jobs=-1, verbose=-1
                )),
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=-1,
        ),
    }


def perform_cross_validation(
    model: BaseEstimator,
    X_train,
    y_train,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Perform k-fold cross-validation on training data using an end-to-end pipeline."""
    pipeline = create_model_pipeline(model)
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

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
