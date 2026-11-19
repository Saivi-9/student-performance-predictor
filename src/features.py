"""
Feature Engineering Module (v2.0)
---------------------------------
Provides both functional and leak-free scikit-learn TransformerMixin feature engineering
adding 8 domain-informed academic, cognitive, and behavioral composite features:
- study_efficiency: hours_studied per tutoring_session
- academic_momentum: weighted blend of prior score, quiz average, and attendance
- sleep_deficit: absolute deviation from optimal 8.0h sleep window
- engagement_score: bounded [0, 1] composite of study, attendance, homework, practice
- study_sleep_balance: sustained study intensity relative to sleep deficit
- digital_distraction_ratio: daily recreational screen time relative to daily study pace
- formative_delta: quiz_average - previous_score trajectory trend
- wellbeing_index: composite mental & physical health balance
"""

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.dataset import EXPECTED_NUMERICAL_COLS, DEFAULT_STUDENT_PROFILE


OPTIMAL_SLEEP_HOURS = 8.0

ENGINEERED_FEATURES = [
    'study_efficiency',
    'academic_momentum',
    'sleep_deficit',
    'engagement_score',
    'study_sleep_balance',
    'digital_distraction_ratio',
    'formative_delta',
    'wellbeing_index',
]


def _fill_with_defaults_or_medians(
    df: pd.DataFrame,
    medians: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Safely impute missing numeric values using training medians or defaults."""
    df = df.copy()
    for col in EXPECTED_NUMERICAL_COLS:
        if col not in df.columns:
            df[col] = DEFAULT_STUDENT_PROFILE.get(col, 0.0)
        if df[col].isnull().any():
            if medians and col in medians:
                fill_val = medians[col]
            else:
                med = df[col].median()
                fill_val = float(med) if pd.notna(med) else float(DEFAULT_STUDENT_PROFILE.get(col, 0.0))
            df[col] = df[col].fillna(fill_val)
    return df


def engineer_features(
    df: pd.DataFrame,
    medians: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add 8 engineered behavioral and academic features to the input DataFrame.

    Args:
        df: Input DataFrame containing raw student features.
        medians: Optional pre-computed training medians to prevent leakage.

    Returns:
        pd.DataFrame: Copy of DataFrame with all 8 engineered feature columns added.
    """
    out = _fill_with_defaults_or_medians(df, medians=medians)

    # 1. Study efficiency: hours studied per tutoring session (smoothed)
    out['study_efficiency'] = out['hours_studied'] / (out['tutoring_sessions'] + 1.0)

    # 2. Academic momentum: multi-signal baseline of prior & current formative performance
    out['academic_momentum'] = (
        0.45 * out['previous_score']
        + 0.35 * out['quiz_average']
        + 0.20 * out['attendance_percent']
    )

    # 3. Sleep deficit: absolute deviation from optimal 8h sleep
    out['sleep_deficit'] = np.abs(out['sleep_hours'] - OPTIMAL_SLEEP_HOURS)

    # 4. Bounded engagement score [0, 1] using fixed domain ceilings (leak-free)
    out['engagement_score'] = np.clip(
        (
            np.clip(out['hours_studied'] / 35.0, 0.0, 1.0)
            + np.clip(out['attendance_percent'] / 100.0, 0.0, 1.0)
            + np.clip(out['assignment_completion_rate'] / 100.0, 0.0, 1.0)
            + np.clip(out['practice_papers_solved'] / 15.0, 0.0, 1.0)
        ) / 4.0,
        0.0,
        1.0,
    )

    # 5. Study-to-sleep balance (higher = focused study without sleep deprivation)
    out['study_sleep_balance'] = out['hours_studied'] / (out['sleep_deficit'] + 1.0)

    # 6. Digital distraction ratio: daily screen hours vs daily study hours
    daily_study = out['hours_studied'] / 7.0
    out['digital_distraction_ratio'] = out['screen_time_hours'] / (daily_study + 0.5)

    # 7. Formative momentum delta: improvement from prior exam to recent quizzes
    out['formative_delta'] = out['quiz_average'] - out['previous_score']

    # 8. Wellbeing index: physical activity balanced against stress and sleep deficit
    out['wellbeing_index'] = (
        (out['physical_activity_hours'] / 10.0)
        - (out['stress_level'] / 10.0)
        - (out['sleep_deficit'] / 5.0)
    )

    return out


class AcademicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer that computes engineered features inside Pipelines.

    Learns numeric medians strictly during fit(X_train) to prevent train/validation leakage.
    """

    def __init__(self):
        self.medians_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None):
        X_df = pd.DataFrame(X).copy()
        self.medians_ = {}
        for col in EXPECTED_NUMERICAL_COLS:
            if col in X_df.columns:
                med = pd.to_numeric(X_df[col], errors='coerce').median()
                self.medians_[col] = float(med) if pd.notna(med) else float(DEFAULT_STUDENT_PROFILE.get(col, 0.0))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        return engineer_features(X_df, medians=self.medians_)
