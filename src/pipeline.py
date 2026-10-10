"""
Pipeline module for constructing leak-free preprocessing and model pipelines.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator


NUMERICAL_FEATURES = [
    'hours_studied',
    'attendance_percent',
    'previous_score',
    'sleep_hours',
    'tutoring_sessions'
]

ORDINAL_FEATURES = ['parental_involvement']
ORDINAL_CATEGORIES = [['Low', 'Medium', 'High']]

NOMINAL_FEATURES = ['extracurricular', 'internet_access']


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer that handles imputation, encoding, and scaling.

    - Numerical features: Impute missing values with median, then standardize (mean=0, std=1).
    - Ordinal features: Impute missing values with most frequent, then ordinal encode ('Low' < 'Medium' < 'High').
    - Nominal features: Impute missing values with most frequent, then one-hot encode.

    Returns:
        ColumnTransformer: Preprocessing transformer.
    """
    # 1. Numerical pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Ordinal categorical pipeline
    ord_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=ORDINAL_CATEGORIES, handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 3. Nominal categorical pipeline
    nom_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # Combine into a unified ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('ord', ord_pipeline, ORDINAL_FEATURES),
            ('nom', nom_pipeline, NOMINAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def create_model_pipeline(model: BaseEstimator) -> Pipeline:
    """Combine the preprocessor and a regressor into an end-to-end Pipeline.

    This ensures that whenever fit() or predict() is called on the pipeline,
    the exact same transformations are applied without data leakage.

    Args:
        model: An instantiated scikit-learn regressor (e.g., LinearRegression, Ridge, RandomForestRegressor).

    Returns:
        Pipeline: End-to-end pipeline ready to fit.
    """
    preprocessor = build_preprocessor()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline
