"""
Hugging Face Pre-Trained Model & Dataset Hub Integration
--------------------------------------------------------
Searches, downloads, caches, benchmarks, and runs live inference against
community pre-trained models on Hugging Face Hub for Student Performance Prediction.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import warnings
import joblib
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# Training-split Z-score statistics for Shalan1/student-performance-predictor-models
_SHALAN1_COLS = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]
_SHALAN1_MEAN = {
    "Hours Studied": 4.983625,
    "Previous Scores": 69.438,
    "Extracurricular Activities": 0.4915,
    "Sleep Hours": 6.52375,
    "Sample Question Papers Practiced": 4.5745,
}
_SHALAN1_STD = {
    "Hours Studied": 2.586911,
    "Previous Scores": 17.358180,
    "Extracurricular Activities": 0.499928,
    "Sleep Hours": 1.695490,
    "Sample Question Papers Practiced": 2.857744,
}


PRETRAINED_CATALOG: List[Dict[str, Any]] = [
    {
        "repo_id": "Shalan1/student-performance-predictor-models",
        "filename": "rf_model.joblib",
        "framework": "scikit-learn (RandomForestRegressor)",
        "task": "Tabular Regression (Performance Index 0-100)",
        "dataset": "10K Student Performance Dataset (5 core features, Z-scored)",
        "supports_live_inference": True,
        "url": "https://huggingface.co/Shalan1/student-performance-predictor-models",
    },
    {
        "repo_id": "SergeyR256/student-performance",
        "filename": "best_model_random_forest.joblib",
        "framework": "scikit-learn Pipeline (RandomForestRegressor)",
        "task": "Tabular Regression (UCI Final Grade G3: 0-20)",
        "dataset": "UCI Portuguese Schools (37 socio-behavioral features)",
        "supports_live_inference": True,
        "url": "https://huggingface.co/SergeyR256/student-performance",
    },
    {
        "repo_id": "sulpha/student_academic_success",
        "filename": "model.pkl",
        "framework": "scikit-learn (RandomForestClassifier)",
        "task": "Multi-class Classification (Dropout / Enrolled / Graduate)",
        "dataset": "Higher Education Student Dropout & Success (36 features)",
        "supports_live_inference": False,
        "url": "https://huggingface.co/sulpha/student_academic_success",
    },
    {
        "repo_id": "nukimayasari/student-performance-model",
        "filename": "student_performance_model.h5",
        "framework": "TensorFlow / Keras (MLP)",
        "task": "Tabular Regression (Exam Grade)",
        "dataset": "Socioeconomic & Academic Factors",
        "supports_live_inference": False,
        "url": "https://huggingface.co/nukimayasari/student-performance-model",
    },
    {
        "repo_id": "olusegvn/student-performance-prediction",
        "filename": "keras_model",
        "framework": "Keras Neural Network",
        "task": "Tabular Regression / Grade Forecasting",
        "dataset": "Academic Assessment Telemetry",
        "supports_live_inference": False,
        "url": "https://huggingface.co/olusegvn/student-performance-prediction",
    },
]


def _patch_legacy_sklearn_imputers(pipeline_obj: Any) -> Any:
    """Backfill `_fill_dtype` on older pickled SimpleImputer steps loaded under newer scikit-learn."""
    preprocessor = getattr(pipeline_obj, "named_steps", {}).get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "transformers_"):
        for _, transformer, _ in preprocessor.transformers_:
            if hasattr(transformer, "named_steps") and "imputer" in transformer.named_steps:
                imp = transformer.named_steps["imputer"]
                if not hasattr(imp, "_fill_dtype") and hasattr(imp, "statistics_"):
                    imp._fill_dtype = imp.statistics_.dtype
    return pipeline_obj


def search_huggingface_models(query: str = "student performance", limit: int = 12) -> List[Dict[str, Any]]:
    """Live-search Hugging Face Hub for pretrained models matching the query."""
    api = HfApi()
    results = []
    for m in api.list_models(search=query, limit=limit):
        results.append({
            "id": m.id,
            "downloads": int(getattr(m, "downloads", 0) or 0),
            "likes": int(getattr(m, "likes", 0) or 0),
            "pipeline_tag": getattr(m, "pipeline_tag", None) or "tabular",
            "tags": [t for t in (m.tags or []) if not t.startswith("region:")][:6],
            "url": f"https://huggingface.co/{m.id}",
        })
    return results


def search_huggingface_datasets(query: str = "student performance", limit: int = 10) -> List[Dict[str, Any]]:
    """Live-search Hugging Face Hub for student performance datasets."""
    api = HfApi()
    results = []
    for d in api.list_datasets(search=query, limit=limit):
        results.append({
            "id": d.id,
            "downloads": int(getattr(d, "downloads", 0) or 0),
            "likes": int(getattr(d, "likes", 0) or 0),
            "tags": [t for t in (d.tags or []) if not t.startswith("region:")][:5],
            "url": f"https://huggingface.co/datasets/{d.id}",
        })
    return results


def load_shalan1_model():
    """Download and load the Shalan1/student-performance-predictor-models Random Forest."""
    path = hf_hub_download("Shalan1/student-performance-predictor-models", "rf_model.joblib")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


def load_sergey_uci_model():
    """Download and load the SergeyR256/student-performance Random Forest Pipeline."""
    path = hf_hub_download("SergeyR256/student-performance", "best_model_random_forest.joblib")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(path)
    return _patch_legacy_sklearn_imputers(model)


def _scale_for_shalan1(df_features: pd.DataFrame) -> pd.DataFrame:
    """Standardize inputs using Shalan1 10K training statistics."""
    scaled = df_features[_SHALAN1_COLS].copy()
    for col in _SHALAN1_COLS:
        scaled[col] = (scaled[col] - _SHALAN1_MEAN[col]) / _SHALAN1_STD[col]
    return scaled


def predict_with_shalan1(
    hours_studied: float,
    previous_score: float,
    extracurricular: str,
    sleep_hours: float,
    practice_papers_solved: float,
) -> float:
    """Run live inference using the Hugging Face Shalan1 pretrained RF model."""
    model = load_shalan1_model()
    scaled_daily_hours = float(np.clip(hours_studied / 3.0, 1.0, 9.0))
    extra_num = 1.0 if str(extracurricular).strip().lower() == "yes" else 0.0
    row = pd.DataFrame([{
        "Hours Studied": scaled_daily_hours,
        "Previous Scores": float(previous_score),
        "Extracurricular Activities": extra_num,
        "Sleep Hours": float(np.clip(sleep_hours, 4.0, 9.0)),
        "Sample Question Papers Practiced": float(np.clip(practice_papers_solved, 0.0, 9.0)),
    }])
    X_scaled = _scale_for_shalan1(row)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = float(model.predict(X_scaled)[0])
    return round(float(np.clip(pred, 0.0, 100.0)), 1)


def _prepare_uci_features_for_sergey(df_uci: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the 7 engineered features expected by SergeyR256/student-performance."""
    df = df_uci.copy()
    df["total_alcohol"] = df["Dalc"] + df["Walc"]
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2.0
    df["goout_studytime_ratio"] = df["goout"] / (df["studytime"] + 1e-5)
    df["alcohol_study_interaction"] = df["total_alcohol"] * df["studytime"]
    df["failures_squared"] = df["failures"] ** 2
    df["age_squared"] = df["age"] ** 2
    df["health_freetime_ratio"] = df["health"] / (df["freetime"] + 1e-5)
    return df


def benchmark_pretrained_models(data_dir: Path | str | None = None) -> List[Dict[str, Any]]:
    """Evaluate downloaded Hugging Face Hub pretrained models on real benchmark datasets."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "external"
    else:
        data_dir = Path(data_dir)

    benchmarks: List[Dict[str, Any]] = []

    # 1. Benchmark Shalan1/student-performance-predictor-models on HF 10K dataset
    try:
        from data.fetch_external import fetch_hf_10k_dataset
        df_10k = fetch_hf_10k_dataset(data_dir).tail(2000).copy()
        df_10k["Extracurricular Activities"] = (
            df_10k["Extracurricular Activities"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(0)
        )
        X_10k_scaled = _scale_for_shalan1(df_10k)
        y_10k = df_10k["Performance Index"]

        shalan_model = load_shalan1_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = shalan_model.predict(X_10k_scaled)

        benchmarks.append({
            "repo_id": "Shalan1/student-performance-predictor-models",
            "model_type": "RandomForestRegressor (Hugging Face Hub)",
            "benchmark_dataset": "10K Student Performance (2,000 holdout rows)",
            "target_scale": "0 - 100 pts",
            "mae": round(float(mean_absolute_error(y_10k, preds)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y_10k, preds))), 3),
            "r2": round(float(r2_score(y_10k, preds)), 4),
            "features_used": 5,
            "url": "https://huggingface.co/Shalan1/student-performance-predictor-models",
        })
    except Exception as exc:
        benchmarks.append({
            "repo_id": "Shalan1/student-performance-predictor-models",
            "error": str(exc),
        })

    # 2. Benchmark SergeyR256/student-performance on real UCI Portuguese dataset
    try:
        from data.fetch_external import fetch_uci_dataset
        df_uci = fetch_uci_dataset(data_dir)
        df_por = df_uci[df_uci["subject"] == "Portuguese"].copy() if "subject" in df_uci.columns else df_uci.copy()
        df_feat = _prepare_uci_features_for_sergey(df_por)
        sergey_model = load_sergey_uci_model()
        expected_cols = list(sergey_model.feature_names_in_)
        X_uci = df_feat[expected_cols]
        y_uci = df_por["G3"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds_uci = sergey_model.predict(X_uci)

        benchmarks.append({
            "repo_id": "SergeyR256/student-performance",
            "model_type": "RandomForest Pipeline (Hugging Face Hub)",
            "benchmark_dataset": "UCI Portuguese Secondary Schools (649 real students)",
            "target_scale": "0 - 20 G3 grade",
            "mae": round(float(mean_absolute_error(y_uci, preds_uci)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y_uci, preds_uci))), 3),
            "r2": round(float(r2_score(y_uci, preds_uci)), 4),
            "features_used": len(expected_cols),
            "url": "https://huggingface.co/SergeyR256/student-performance",
        })
    except Exception as exc:
        benchmarks.append({
            "repo_id": "SergeyR256/student-performance",
            "error": str(exc),
        })

    return benchmarks
