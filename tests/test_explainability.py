"""
Unit tests for the explainability module (v2.0).
"""

import pandas as pd
from sklearn.linear_model import Ridge
from data.generate_dataset import generate_student_data
from src.dataset import split_data, DEFAULT_STUDENT_PROFILE
from src.pipeline import create_model_pipeline
from src.explainability import (
    compute_permutation_importance,
    plot_feature_importance,
    simulate_what_if_trajectory,
)


def _make_dummy_dataset():
    df = generate_student_data(n_samples=120, random_seed=42)
    X_train, _, _, y_train, _, _ = split_data(df, test_size=0.15, val_size=0.15, random_state=42)
    return X_train, y_train


def test_compute_permutation_importance_structure():
    X, y = _make_dummy_dataset()
    pipeline = create_model_pipeline(Ridge(alpha=1.0, random_state=42))
    pipeline.fit(X, y)

    importance_df = compute_permutation_importance(pipeline, X, y, n_repeats=3, random_state=42)

    assert isinstance(importance_df, pd.DataFrame)
    assert set(importance_df.columns) == {"feature", "importance_mean", "importance_std"}
    assert len(importance_df) == len(X.columns)
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

    base_student = DEFAULT_STUDENT_PROFILE.copy()
    hours = [2.0, 5.0, 10.0, 18.0]
    traj = simulate_what_if_trajectory(pipeline, base_student, 'hours_studied', hours)

    assert len(traj) == 4
    assert 'hours_studied' in traj.columns
    assert 'predicted_score' in traj.columns
    scores = traj['predicted_score'].tolist()
    assert scores[-1] > scores[0]
