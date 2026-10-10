"""
Data Generation Script for Student Performance Dataset.

Generates a realistic tabular dataset of student academic habits and final exam scores.
Includes both numerical and categorical features, as well as a small percentage of
naturally occurring missing values to demonstrate real-world data cleaning and imputation.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_student_data(n_samples: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """Generate synthetic student performance data with realistic relationships.

    Args:
        n_samples: Number of student records to generate.
        random_seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: The generated dataset.
    """
    np.random.seed(random_seed)

    # Numerical features
    hours_studied = np.clip(np.random.normal(loc=5.5, scale=2.0, size=n_samples), 1.0, 12.0)
    attendance_percent = np.clip(np.random.normal(loc=82.0, scale=10.0, size=n_samples), 50.0, 100.0)
    previous_score = np.clip(np.random.normal(loc=70.0, scale=12.0, size=n_samples), 35.0, 98.0)
    sleep_hours = np.clip(np.random.normal(loc=7.0, scale=1.2, size=n_samples), 4.0, 10.0)
    tutoring_sessions = np.random.poisson(lam=2.0, size=n_samples)
    tutoring_sessions = np.clip(tutoring_sessions, 0, 8)

    # Categorical features
    parental_involvement = np.random.choice(
        ['Low', 'Medium', 'High'],
        size=n_samples,
        p=[0.25, 0.50, 0.25]
    )
    extracurricular = np.random.choice(['No', 'Yes'], size=n_samples, p=[0.45, 0.55])
    internet_access = np.random.choice(['No', 'Yes'], size=n_samples, p=[0.15, 0.85])

    # Parental impact mapping
    parental_boost = {'Low': 0.0, 'Medium': 2.5, 'High': 5.0}
    parental_numeric = np.array([parental_boost[p] for p in parental_involvement])

    extra_boost = np.where(extracurricular == 'Yes', 1.8, 0.0)
    internet_boost = np.where(internet_access == 'Yes', 2.2, 0.0)

    # Realistic linear combination with stochastic Gaussian noise
    noise = np.random.normal(loc=0.0, scale=3.2, size=n_samples)
    raw_score = (
        10.0
        + (2.6 * hours_studied)
        + (0.28 * attendance_percent)
        + (0.36 * previous_score)
        + (0.75 * sleep_hours)
        + (1.1 * tutoring_sessions)
        + parental_numeric
        + extra_boost
        + internet_boost
        + noise
    )

    final_score = np.clip(np.round(raw_score, 1), 20.0, 100.0)

    df = pd.DataFrame({
        'hours_studied': np.round(hours_studied, 1),
        'attendance_percent': np.round(attendance_percent, 1),
        'previous_score': np.round(previous_score, 1),
        'sleep_hours': np.round(sleep_hours, 1),
        'tutoring_sessions': tutoring_sessions,
        'parental_involvement': parental_involvement,
        'extracurricular': extracurricular,
        'internet_access': internet_access,
        'final_score': final_score,
    })

    # Introduce realistic missing values (~2% in sleep_hours and tutoring_sessions)
    missing_sleep_idx = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    missing_tutoring_idx = np.random.choice(n_samples, size=int(0.015 * n_samples), replace=False)
    missing_parental_idx = np.random.choice(n_samples, size=int(0.01 * n_samples), replace=False)

    df.loc[missing_sleep_idx, 'sleep_hours'] = np.nan
    df.loc[missing_tutoring_idx, 'tutoring_sessions'] = np.nan
    df.loc[missing_parental_idx, 'parental_involvement'] = np.nan

    return df


if __name__ == '__main__':
    data_dir = Path(__file__).resolve().parent
    output_path = data_dir / 'student_data.csv'
    data_dir.mkdir(parents=True, exist_ok=True)

    df = generate_student_data(n_samples=1000, random_seed=42)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} student records saved to {output_path}")
