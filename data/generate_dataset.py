"""
Data Generation Script for Multi-Dimensional Student Performance Dataset (v2.0).

Generates a 20,000-record tabular dataset with 20 diverse raw features spanning:
- Academic effort & formative metrics (hours_studied, attendance_percent, previous_score,
  quiz_average, assignment_completion_rate, practice_papers_solved, tutoring_sessions)
- Cognitive & wellbeing telemetry (sleep_hours, physical_activity_hours,
  screen_time_hours, stress_level, commute_minutes)
- Socioeconomic & contextual factors (parental_involvement, motivation_level,
  access_to_resources, parental_education, extracurricular, internet_access,
  school_type, learning_style)

Includes realistic non-linear relationships, interaction effects, and ~1.5% missingness
to exercise leak-free imputation and feature engineering pipelines.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_student_data(n_samples: int = 20000, random_seed: int = 42) -> pd.DataFrame:
    """Generate a multi-dimensional student performance dataset.

    Args:
        n_samples: Number of student records to generate (default: 20,000).
        random_seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Generated dataset with 20 raw features + final_score.
    """
    rng = np.random.default_rng(random_seed)

    # Latent student aptitude & diligence factors for realistic cross-feature correlation
    aptitude = rng.normal(0.0, 1.0, size=n_samples)
    diligence = rng.normal(0.0, 1.0, size=n_samples)

    # 1. Academic & Study Numerical Features (12 features)
    hours_studied = np.clip(
        11.0 + 4.5 * diligence + rng.normal(0, 3.2, n_samples),
        1.0, 35.0
    )
    attendance_percent = np.clip(
        82.0 + 6.5 * diligence + rng.normal(0, 8.0, n_samples),
        40.0, 100.0
    )
    previous_score = np.clip(
        70.0 + 9.0 * aptitude + 3.0 * diligence + rng.normal(0, 7.5, n_samples),
        25.0, 99.0
    )
    quiz_average = np.clip(
        0.72 * previous_score + 0.20 * attendance_percent + 0.5 * hours_studied + rng.normal(0, 5.5, n_samples),
        25.0, 100.0
    )
    assignment_completion_rate = np.clip(
        76.0 + 9.5 * diligence + 2.0 * aptitude + rng.normal(0, 9.0, n_samples),
        30.0, 100.0
    )
    practice_papers_solved = np.clip(
        np.round(4.0 + 2.0 * diligence + rng.poisson(1.8, n_samples)),
        0, 15
    ).astype(float)
    sleep_hours = np.clip(
        rng.normal(7.1, 1.25, n_samples) - 0.15 * np.maximum(hours_studied - 18.0, 0),
        3.5, 11.0
    )
    tutoring_sessions = np.clip(
        rng.poisson(2.2, n_samples),
        0, 12
    ).astype(float)
    physical_activity_hours = np.clip(
        rng.normal(4.2, 2.1, n_samples),
        0.0, 14.0
    )
    screen_time_hours = np.clip(
        rng.normal(3.6, 1.6, n_samples) - 0.45 * diligence,
        0.5, 10.0
    )
    stress_level = np.clip(
        np.round(5.2 - 0.5 * (sleep_hours - 7.0) - 0.3 * physical_activity_hours + 0.12 * np.maximum(hours_studied - 16, 0) + rng.normal(0, 1.6, n_samples)),
        1, 10
    ).astype(float)
    commute_minutes = np.clip(
        np.round(rng.gamma(shape=2.5, scale=12.0, size=n_samples)),
        5, 120
    ).astype(float)

    # 2. Ordinal Categorical Features (4 features)
    parental_involvement = rng.choice(['Low', 'Medium', 'High'], size=n_samples, p=[0.22, 0.50, 0.28])
    motivation_level = rng.choice(['Low', 'Medium', 'High'], size=n_samples, p=[0.24, 0.48, 0.28])
    access_to_resources = rng.choice(['Low', 'Medium', 'High'], size=n_samples, p=[0.20, 0.48, 0.32])
    parental_education = rng.choice(['High School', 'Bachelor', 'Master+'], size=n_samples, p=[0.35, 0.42, 0.23])

    # 3. Nominal Categorical Features (4 features)
    extracurricular = rng.choice(['Yes', 'No'], size=n_samples, p=[0.58, 0.42])
    internet_access = rng.choice(['Yes', 'No'], size=n_samples, p=[0.88, 0.12])
    school_type = rng.choice(['Public', 'Private'], size=n_samples, p=[0.70, 0.30])
    learning_style = rng.choice(
        ['Visual', 'Auditory', 'Kinesthetic', 'Reading/Writing'],
        size=n_samples,
        p=[0.32, 0.22, 0.22, 0.24]
    )

    # Numeric effect mappings for target synthesis
    ord_map_3 = {'Low': 0.0, 'Medium': 2.2, 'High': 4.6}
    edu_map = {'High School': 0.0, 'Bachelor': 1.8, 'Master+': 3.5}

    p_inv_boost = np.array([ord_map_3[v] for v in parental_involvement])
    mot_boost = np.array([ord_map_3[v] for v in motivation_level])
    res_boost = np.array([ord_map_3[v] * 0.7 for v in access_to_resources])
    edu_boost = np.array([edu_map[v] for v in parental_education])
    extra_boost = np.where(extracurricular == 'Yes', 1.4, 0.0)
    net_boost = np.where(internet_access == 'Yes', 2.0, 0.0)
    school_boost = np.where(school_type == 'Private', 1.1, 0.0)

    # Non-linear biological & educational effects
    study_effect = 1.65 * np.sqrt(hours_studied * 4.5) + 0.35 * hours_studied
    sleep_penalty = 1.15 * np.abs(sleep_hours - 8.0)
    stress_penalty = 0.85 * np.maximum(stress_level - 6.0, 0.0) ** 1.3
    screen_penalty = 0.65 * np.maximum(screen_time_hours - 3.5, 0.0)
    commute_penalty = 0.04 * np.maximum(commute_minutes - 45.0, 0.0)
    synergy_bonus = 0.0018 * (attendance_percent * assignment_completion_rate)

    noise = rng.normal(0.0, 2.35, size=n_samples)

    raw_score = (
        -24.5
        + 0.34 * previous_score
        + 0.26 * quiz_average
        + study_effect
        + 0.14 * attendance_percent
        + 0.10 * assignment_completion_rate
        + 0.72 * practice_papers_solved
        + 0.68 * tutoring_sessions
        + 0.30 * physical_activity_hours
        + synergy_bonus
        + p_inv_boost
        + mot_boost
        + res_boost
        + edu_boost
        + extra_boost
        + net_boost
        + school_boost
        - sleep_penalty
        - stress_penalty
        - screen_penalty
        - commute_penalty
        + noise
    )

    final_score = np.clip(np.round(raw_score, 1), 15.0, 100.0)

    df = pd.DataFrame({
        'hours_studied': np.round(hours_studied, 1),
        'attendance_percent': np.round(attendance_percent, 1),
        'previous_score': np.round(previous_score, 1),
        'quiz_average': np.round(quiz_average, 1),
        'assignment_completion_rate': np.round(assignment_completion_rate, 1),
        'practice_papers_solved': practice_papers_solved,
        'sleep_hours': np.round(sleep_hours, 1),
        'tutoring_sessions': tutoring_sessions,
        'physical_activity_hours': np.round(physical_activity_hours, 1),
        'screen_time_hours': np.round(screen_time_hours, 1),
        'stress_level': stress_level,
        'commute_minutes': commute_minutes,
        'parental_involvement': parental_involvement,
        'motivation_level': motivation_level,
        'access_to_resources': access_to_resources,
        'parental_education': parental_education,
        'extracurricular': extracurricular,
        'internet_access': internet_access,
        'school_type': school_type,
        'learning_style': learning_style,
        'final_score': final_score,
    })

    # Introduce realistic ~1.5% missingness across selected columns
    for col, frac in [
        ('sleep_hours', 0.018),
        ('tutoring_sessions', 0.015),
        ('screen_time_hours', 0.014),
        ('parental_involvement', 0.012),
        ('parental_education', 0.012),
        ('commute_minutes', 0.010),
    ]:
        miss_idx = rng.choice(n_samples, size=int(frac * n_samples), replace=False)
        df.loc[miss_idx, col] = np.nan

    return df


if __name__ == '__main__':
    data_dir = Path(__file__).resolve().parent
    output_path = data_dir / 'student_data.csv'
    data_dir.mkdir(parents=True, exist_ok=True)

    df = generate_student_data(n_samples=20000, random_seed=42)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} student records ({df.shape[1]} columns) saved to {output_path}")
