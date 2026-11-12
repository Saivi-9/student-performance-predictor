"""
Dataset module for loading, validating, and splitting multi-dimensional student performance data.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split


EXPECTED_NUMERICAL_COLS = [
    'hours_studied',
    'attendance_percent',
    'previous_score',
    'quiz_average',
    'assignment_completion_rate',
    'practice_papers_solved',
    'sleep_hours',
    'tutoring_sessions',
    'physical_activity_hours',
    'screen_time_hours',
    'stress_level',
    'commute_minutes',
]

EXPECTED_ORDINAL_COLS = [
    'parental_involvement',
    'motivation_level',
    'access_to_resources',
    'parental_education',
]

EXPECTED_NOMINAL_COLS = [
    'extracurricular',
    'internet_access',
    'school_type',
    'learning_style',
]

EXPECTED_CATEGORICAL_COLS = EXPECTED_ORDINAL_COLS + EXPECTED_NOMINAL_COLS

TARGET_COL = 'final_score'

DEFAULT_STUDENT_PROFILE: Dict[str, Any] = {
    'hours_studied': 12.0,
    'attendance_percent': 88.0,
    'previous_score': 74.0,
    'quiz_average': 78.0,
    'assignment_completion_rate': 85.0,
    'practice_papers_solved': 5,
    'sleep_hours': 7.5,
    'tutoring_sessions': 2,
    'physical_activity_hours': 4.5,
    'screen_time_hours': 3.0,
    'stress_level': 4,
    'commute_minutes': 25,
    'parental_involvement': 'Medium',
    'motivation_level': 'Medium',
    'access_to_resources': 'Medium',
    'parental_education': 'Bachelor',
    'extracurricular': 'Yes',
    'internet_access': 'Yes',
    'school_type': 'Public',
    'learning_style': 'Visual',
}


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load the student dataset from a CSV file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect schema, validate missing values, and check domain constraints."""
    missing_counts = df.isnull().sum().to_dict()
    missing_pct = (df.isnull().sum() / max(len(df), 1) * 100).round(2).to_dict()

    all_expected = EXPECTED_NUMERICAL_COLS + EXPECTED_CATEGORICAL_COLS + [TARGET_COL]
    missing_columns = [col for col in all_expected if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    target_missing = df[TARGET_COL].isnull().sum()

    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_counts': missing_counts,
        'missing_pct': missing_pct,
        'target_missing_count': int(target_missing),
        'numerical_columns': EXPECTED_NUMERICAL_COLS,
        'categorical_columns': EXPECTED_CATEGORICAL_COLS,
        'target_column': TARGET_COL,
    }


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split dataset into Train, Validation, and Test partitions.

    Adheres to Strict Featurization Ordering: Always split data into subsets
    BEFORE fitting any preprocessing transformers or scalers to avoid data leakage.
    """
    clean_df = df.dropna(subset=[TARGET_COL]).copy()

    feature_cols = EXPECTED_NUMERICAL_COLS + EXPECTED_CATEGORICAL_COLS
    X = clean_df[feature_cols]
    y = clean_df[TARGET_COL]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
