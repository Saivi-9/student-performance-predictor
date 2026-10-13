"""
Feature Engineering Module
--------------------------
Adds informative derived features to improve model performance:
- study_efficiency: hours_studied per tutoring_session (effort density)
- academic_momentum: weighted combination of previous_score + attendance
- sleep_deficit: deviation from optimal 8h sleep
- engagement_score: composite of study hours, attendance, and tutoring
"""

from typing import Tuple
import pandas as pd
import numpy as np


OPTIMAL_SLEEP_HOURS = 8.0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 4 engineered features that capture non-linear academic dynamics.

    Args:
        df: Input DataFrame with original columns.

    Returns:
        pd.DataFrame: DataFrame with additional engineered feature columns.
    """
    df = df.copy()

    # Fill missing values with column medians for robust feature engineering
    for col in ['hours_studied', 'attendance_percent', 'previous_score', 'sleep_hours', 'tutoring_sessions']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 1. Study efficiency: hours studied per tutoring session (with smoothing)
    df['study_efficiency'] = df['hours_studied'] / (df['tutoring_sessions'] + 1.0)

    # 2. Academic momentum: weighted combination of prior score and attendance
    df['academic_momentum'] = (
        0.65 * df['previous_score'] +
        0.35 * df['attendance_percent']
    )

    # 3. Sleep deficit: absolute deviation from optimal 8h sleep (penalises both under/over-sleep)
    df['sleep_deficit'] = np.abs(df['sleep_hours'] - OPTIMAL_SLEEP_HOURS)

    # 4. Overall engagement score: standardised composite of 3 behavioural signals
    max_hours = max(float(df['hours_studied'].max()), 1.0)
    max_sessions = max(float(df['tutoring_sessions'].max()), 1.0)
    raw_engagement = (
        df['hours_studied'] / max_hours +
        df['attendance_percent'] / 100.0 +
        df['tutoring_sessions'] / max_sessions
    ) / 3.0
    df['engagement_score'] = raw_engagement

    return df


ENGINEERED_FEATURES = [
    'study_efficiency',
    'academic_momentum',
    'sleep_deficit',
    'engagement_score',
]
