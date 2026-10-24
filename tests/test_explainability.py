"""
Unit tests for the explainability module.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from src.pipeline import create_model_pipeline
from src.explainability import (
    compute_permutation_importance,
    plot_feature_importance,
    simulate_what_if_trajectory,
)


def _make_dummy_dataset():
    np.random.seed(42)
    n = 100
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
        0.4 * df['previous_score'] +
        1.5 * df['hours_studied'] +
        0.2 * df['attendance_percent'] +
        np.random.normal(0, 2, size=n)
    )
    return df, y


def test_compute_permutation_importance_structure():
    X, y = _make_dummy_dataset()
    pipeline = create_model_pipeline(Ridge(alpha=1.0, random_state=42))
    pipeline.fit(X, y)

    importance_df = compute_permutation_importance(pipeline, X, y, n_repeats=3, random_state=42)

    assert isinstance(importance_df, pd.DataFrame)
    assert set(importance_df.columns) == {"feature", "importance_mean", "importance_std"}
    assert len(importance_df) == len(X.columns)
    # Check that it's sorted descending
    assert (importance_df["importance_mean"].diff().dropna() <= 0).all()


def test_plot_feature_importance_saves_file(tmp_path):
    importance_df = pd.DataFrame({
        "feature": ["previous_score", "hours_studied", "attendance_percent"],
        "importance_mean": [0.45, 0.32, 0.15],
        "importance_std": [0.03, 0.02, 0.01],
    })

    out_file = tmp_path / "importance.png"
    plot_feature_importance(importance_df, output_path=out_file)

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_simulate_what_if_trajectory():
    X, y = _make_dummy_dataset()
    pipeline = create_model_pipeline(Ridge(alpha=1.0, random_state=42))
    pipeline.fit(X, y)

    base_student = {
        'hours_studied': 5.0,
        'attendance_percent': 80.0,
        'previous_score': 70.0,
        'sleep_hours': 7.0,
        'tutoring_sessions': 1,
        'parental_involvement': 'Medium',
        'extracurricular': 'Yes',
        'internet_access': 'Yes'
    }

    hours = [2.0, 5.0, 10.0, 15.0]
    traj = simulate_what_if_trajectory(pipeline, base_student, 'hours_studied', hours)

    assert len(traj) == 4
    assert 'hours_studied' in traj.columns
    assert 'predicted_score' in traj.columns
    # With positive weight on hours_studied, higher study hours should yield higher scores
    scores = traj['predicted_score'].tolist()
    assert scores[-1] > scores[0]
