"""
Pipeline module for constructing leak-free feature engineering, preprocessing, and model pipelines.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator

from src.dataset import (
    EXPECTED_NUMERICAL_COLS,
    EXPECTED_ORDINAL_COLS,
    EXPECTED_NOMINAL_COLS,
)
from src.features import AcademicFeatureEngineer, ENGINEERED_FEATURES


NUMERICAL_FEATURES = EXPECTED_NUMERICAL_COLS + ENGINEERED_FEATURES

ORDINAL_FEATURES = EXPECTED_ORDINAL_COLS
ORDINAL_CATEGORIES = [
    ['Low', 'Medium', 'High'],                 # parental_involvement
    ['Low', 'Medium', 'High'],                 # motivation_level
    ['Low', 'Medium', 'High'],                 # access_to_resources
    ['High School', 'Bachelor', 'Master+'],    # parental_education
]

NOMINAL_FEATURES = EXPECTED_NOMINAL_COLS


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer that handles imputation, encoding, and scaling
    across all 20 raw and 8 engineered features (28 features total).
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    ord_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])

    nom_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('ord', ord_pipeline, ORDINAL_FEATURES),
            ('nom', nom_pipeline, NOMINAL_FEATURES),
        ],
        remainder='drop',
    )


def create_model_pipeline(model: BaseEstimator) -> Pipeline:
    """Combine AcademicFeatureEngineer, ColumnTransformer, and a regressor into an end-to-end Pipeline.

    This guarantees zero data leakage and zero train-serving skew across all 28 features.
    """
    return Pipeline([
        ('feature_engineer', AcademicFeatureEngineer()),
        ('preprocessor', build_preprocessor()),
        ('model', model),
    ])
