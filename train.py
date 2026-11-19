"""
Main Training, Benchmarking, Explainability, and Model Registry Script (v2.0).

Executes the end-to-end ML workflow:
1. Loads or generates the 20,000-row, 20-feature student dataset (and caches external UCI/HF datasets).
2. Validates schema and missing values.
3. Splits into Train (70%), Validation (15%), and Test (15%) partitions before fitting.
4. Benchmarks 9 diverse model architectures (Linear, Ridge, ElasticNet, Random Forest,
   Extra Trees, HistGradientBoosting, LightGBM, MLP Neural Network, Stacking Ensemble).
5. Saves every trained model to saved_models/registry/ and champion to saved_models/best_model.joblib.
6. Computes permutation feature importance & diagnostic scatter samples.
7. Persists structured metrics to reports/benchmark_results.json and diagnostic plots.
"""

import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from data.generate_dataset import generate_student_data
from data.fetch_external import fetch_all_external_datasets
from src.dataset import (
    load_data,
    validate_data,
    split_data,
    EXPECTED_NUMERICAL_COLS,
    EXPECTED_CATEGORICAL_COLS,
)
from src.features import ENGINEERED_FEATURES
from src.pipeline import create_model_pipeline
from src.models import get_candidate_models, perform_cross_validation, model_slug
from src.evaluate import (
    compute_metrics,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_model_comparison,
)
from src.explainability import compute_permutation_importance, plot_feature_importance


def run_pipeline(
    n_samples: int = 20000,
    cv_folds: int = 5,
    fetch_external: bool = True,
    force_regenerate: bool = False,
):
    project_root = Path(__file__).resolve().parent
    data_path = project_root / 'data' / 'student_data.csv'
    models_dir = project_root / 'saved_models'
    registry_dir = models_dir / 'registry'
    reports_dir = project_root / 'reports'
    figures_dir = reports_dir / 'figures'

    models_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("   STUDENT PERFORMANCE PREDICTOR v2.0 - MULTI-MODEL ML TOURNAMENT PIPELINE   ")
    print("=" * 78)

    # 1. Dataset loading / generation
    if force_regenerate or not data_path.exists():
        print(f"Generating {n_samples:,}-row multi-dimensional student dataset at {data_path}...")
        df = generate_student_data(n_samples=n_samples, random_seed=42)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = load_data(data_path)
        if len(df) < 5000 or len(df.columns) < 15:
            print(f"Existing dataset has {len(df)} rows & {len(df.columns)} cols; upgrading to {n_samples:,} rows & 21 cols...")
            df = generate_student_data(n_samples=n_samples, random_seed=42)
            df.to_csv(data_path, index=False)
        else:
            print(f"Loaded dataset from {data_path} ({len(df):,} rows, {df.shape[1]} columns)")

    if fetch_external:
        ext_datasets = fetch_all_external_datasets(project_root / 'data' / 'external')
        for ext_name, ext_df in ext_datasets.items():
            print(f"  [External Benchmark Ready] '{ext_name}': {ext_df.shape[0]:,} rows x {ext_df.shape[1]} cols")

    # 2. Validation
    validation_summary = validate_data(df)
    print(
        f"\n[Data Summary] Rows: {validation_summary['total_rows']:,} | "
        f"Raw Features: {len(EXPECTED_NUMERICAL_COLS) + len(EXPECTED_CATEGORICAL_COLS)} | "
        f"Engineered Features: {len(ENGINEERED_FEATURES)} | "
        f"Total Pipeline Features: {len(EXPECTED_NUMERICAL_COLS) + len(EXPECTED_CATEGORICAL_COLS) + len(ENGINEERED_FEATURES)}"
    )
    missing_cols = {k: v for k, v in validation_summary['missing_counts'].items() if v > 0}
    if missing_cols:
        print("[Missing Values Handled via Leak-Free Imputation]:")
        for col, count in missing_cols.items():
            print(f"  - {col}: {count} missing ({validation_summary['missing_pct'][col]}%)")

    # 3. Train/Val/Test Split (Strict featurization ordering)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, test_size=0.15, val_size=0.15, random_state=42
    )
    print(f"\n[Strict Split] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # 4. 9-Model Tournament Benchmarking & Cross-Validation
    candidate_models = get_candidate_models(random_state=42)
    benchmark_records = []
    fitted_pipelines = {}

    print(f"\n[Tournament] Benchmarking {len(candidate_models)} architectures ({cv_folds}-Fold CV)...")
    for name, model in candidate_models.items():
        t0 = time.perf_counter()
        cv_res = perform_cross_validation(model, X_train, y_train, cv=cv_folds, random_state=42)

        pipeline = create_model_pipeline(model)
        pipeline.fit(X_train, y_train)
        elapsed = round(time.perf_counter() - t0, 2)

        fitted_pipelines[name] = pipeline
        slug = model_slug(name)
        joblib.dump(pipeline, registry_dir / f"{slug}.joblib")

        val_metrics = compute_metrics(y_val, pipeline.predict(X_val))
        test_metrics = compute_metrics(y_test, pipeline.predict(X_test))

        print(
            f"  {name:22s} | CV RMSE: {cv_res['cv_rmse_mean']:.3f} | "
            f"Val RMSE: {val_metrics['RMSE']:.3f} | Test R²: {test_metrics['R2']:.4f} | {elapsed:.1f}s"
        )

        benchmark_records.append({
            'Model': name,
            'slug': slug,
            'CV_MAE': round(cv_res['cv_mae_mean'], 3),
            'CV_RMSE': round(cv_res['cv_rmse_mean'], 3),
            'CV_R2': round(cv_res['cv_r2_mean'], 4),
            'Val_MAE': val_metrics['MAE'],
            'Val_RMSE': val_metrics['RMSE'],
            'Val_R2': val_metrics['R2'],
            'Test_MAE': test_metrics['MAE'],
            'Test_RMSE': test_metrics['RMSE'],
            'Test_R2': test_metrics['R2'],
            'RMSE': val_metrics['RMSE'],
            'R2': val_metrics['R2'],
            'train_time_sec': elapsed,
        })

    benchmark_df = pd.DataFrame(benchmark_records).sort_values(by='Val_RMSE', ascending=True).reset_index(drop=True)
    print("\n[Tournament Leaderboard (Sorted by Validation RMSE)]:")
    print(benchmark_df[['Model', 'CV_RMSE', 'Val_RMSE', 'Val_R2', 'Test_RMSE', 'Test_R2']].to_string(index=False))

    # 5. Champion Selection
    best_record = benchmark_df.iloc[0]
    best_model_name = best_record['Model']
    best_slug = best_record['slug']
    champion_pipeline = fitted_pipelines[best_model_name]
    print(f"\n🏆 Champion Selected: {best_model_name} (Val RMSE: {best_record['Val_RMSE']:.3f}, Test R²: {best_record['Test_R2']:.4f})")

    # 6. Final Holdout Test Evaluation
    y_test_pred = champion_pipeline.predict(X_test)
    champion_test_metrics = compute_metrics(y_test, y_test_pred)

    # Save champion artifact
    model_save_path = models_dir / 'best_model.joblib'
    joblib.dump(champion_pipeline, model_save_path)
    print(f"[Saved Champion] {model_save_path}")

    # 7. Permutation Feature Importance on Holdout Validation Set
    print("[Explainability] Computing permutation feature importance on validation holdout...")
    importance_df = compute_permutation_importance(
        champion_pipeline, X_val, y_val, n_repeats=5, random_state=42
    )

    # 8. Generate Visual Artifacts
    plot_model_comparison(benchmark_df, figures_dir / 'model_comparison.png')
    plot_actual_vs_predicted(y_test, y_test_pred, best_model_name, figures_dir / 'actual_vs_predicted.png')
    plot_residuals(y_test, y_test_pred, best_model_name, figures_dir / 'residual_plot.png')
    plot_feature_importance(importance_df, figures_dir / 'feature_importance.png', top_n=12)

    # 9. Export Structured JSON Metadata for Real-Time API & Frontend Dashboard
    sample_idx = np.linspace(0, len(y_test) - 1, min(160, len(y_test)), dtype=int)
    y_test_arr = np.array(y_test)
    diagnostic_samples = [
        {
            'actual': round(float(y_test_arr[i]), 1),
            'predicted': round(float(np.clip(y_test_pred[i], 0, 100)), 1),
            'residual': round(float(y_test_arr[i] - y_test_pred[i]), 2),
        }
        for i in sample_idx
    ]

    model_entries = []
    for _, row in benchmark_df.iterrows():
        model_entries.append({
            'name': row['Model'],
            'slug': row['slug'],
            'cv_mae': float(row['CV_MAE']),
            'cv_rmse': float(row['CV_RMSE']),
            'cv_r2': float(row['CV_R2']),
            'val_mae': float(row['Val_MAE']),
            'val_rmse': float(row['Val_RMSE']),
            'val_r2': float(row['Val_R2']),
            'test_mae': float(row['Test_MAE']),
            'test_rmse': float(row['Test_RMSE']),
            'test_r2': float(row['Test_R2']),
            'train_time_sec': float(row['train_time_sec']),
            'is_champion': bool(row['Model'] == best_model_name),
        })

    numeric_stats = {}
    for col in EXPECTED_NUMERICAL_COLS:
        numeric_stats[col] = {
            'mean': round(float(df[col].mean()), 2),
            'std': round(float(df[col].std()), 2),
            'min': round(float(df[col].min()), 2),
            'max': round(float(df[col].max()), 2),
        }

    report_payload = {
        'version': '2.0.0',
        'champion_model': best_model_name,
        'champion_slug': best_slug,
        'test_metrics': champion_test_metrics,
        'dataset_summary': {
            'total_rows': int(len(df)),
            'train_rows': int(len(X_train)),
            'val_rows': int(len(X_val)),
            'test_rows': int(len(X_test)),
            'raw_features': len(EXPECTED_NUMERICAL_COLS) + len(EXPECTED_CATEGORICAL_COLS),
            'engineered_features': len(ENGINEERED_FEATURES),
            'total_pipeline_features': len(EXPECTED_NUMERICAL_COLS) + len(EXPECTED_CATEGORICAL_COLS) + len(ENGINEERED_FEATURES),
            'engineered_feature_names': ENGINEERED_FEATURES,
            'missing_values_imputed': int(sum(missing_cols.values())),
            'score_mean': round(float(df['final_score'].mean()), 2),
            'score_std': round(float(df['final_score'].std()), 2),
            'numeric_stats': numeric_stats,
        },
        'models': model_entries,
        'feature_importances': importance_df.to_dict(orient='records'),
        'diagnostic_samples': diagnostic_samples,
    }

    report_path = reports_dir / 'benchmark_results.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_payload, f, indent=2)
    print(f"[Saved Report] {report_path}")

    print("\n" + "=" * 78)
    print("                     TRAINING TOURNAMENT COMPLETE!                           ")
    print("=" * 78)
    return champion_pipeline, champion_test_metrics


if __name__ == '__main__':
    run_pipeline()
