"""
Dataset module for loading, validating, and splitting student performance data.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split


EXPECTED_NUMERICAL_COLS = [
    'hours_studied',
    'attendance_percent',
    'previous_score',
    'sleep_hours',
    'tutoring_sessions'
]

EXPECTED_CATEGORICAL_COLS = [
    'parental_involvement',
    'extracurricular',
    'internet_access'
]

TARGET_COL = 'final_score'


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load the student dataset from a CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    return df


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect schema, validate missing values, and check domain constraints.

    Args:
        df: Input DataFrame.

    Returns:
        Dict[str, Any]: Summary of validation status and diagnostics.
    """
    missing_counts = df.isnull().sum().to_dict()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2).to_dict()

    # Verify column existence
    all_expected = EXPECTED_NUMERICAL_COLS + EXPECTED_CATEGORICAL_COLS + [TARGET_COL]
    missing_columns = [col for col in all_expected if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    # Check for rows with missing target values (drop them if any exist)
    target_missing = df[TARGET_COL].isnull().sum()

    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_counts': missing_counts,
        'missing_pct': missing_pct,
        'target_missing_count': int(target_missing),
        'numerical_columns': EXPECTED_NUMERICAL_COLS,
        'categorical_columns': EXPECTED_CATEGORICAL_COLS,
        'target_column': TARGET_COL
    }
    return summary


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split dataset into Train, Validation, and Test partitions.

    Adheres to Strict Featurization Ordering: Always split data into subsets
    BEFORE fitting any preprocessing transformers or scalers to avoid data leakage.

    Args:
        df: Input DataFrame.
        test_size: Proportion of test set (e.g. 0.15).
        val_size: Proportion of validation set (e.g. 0.15).
        random_state: Random state seed.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Clean rows missing the target variable
    clean_df = df.dropna(subset=[TARGET_COL]).copy()

    feature_cols = EXPECTED_NUMERICAL_COLS + EXPECTED_CATEGORICAL_COLS
    X = clean_df[feature_cols]
    y = clean_df[TARGET_COL]

    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Then split the remaining into train and validation
    # Calculate adjusted val_size relative to remaining temp dataset
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
