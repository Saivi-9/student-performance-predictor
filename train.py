"""
Main Training and Model Selection Script.

Executes the end-to-end ML workflow:
1. Loads or generates dataset.
2. Validates schema and missing values.
3. Splits into Train, Validation, and Test partitions.
4. Benchmarks candidate models using k-fold cross-validation.
5. Evaluates candidate models on the validation set.
6. Selects the champion model and evaluates on the unseen test set.
7. Saves the trained pipeline and generates visualization artifacts.
"""

from pathlib import Path
import joblib
import pandas as pd
from data.generate_dataset import generate_student_data
from src.dataset import load_data, validate_data, split_data
from src.pipeline import create_model_pipeline
from src.models import get_candidate_models, perform_cross_validation
from src.evaluate import (
    compute_metrics,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_model_comparison
)


def run_pipeline():
    project_root = Path(__file__).resolve().parent
    data_path = project_root / 'data' / 'student_data.csv'
    models_dir = project_root / 'saved_models'
    figures_dir = project_root / 'reports' / 'figures'

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      STUDENT PERFORMANCE PREDICTOR - MACHINE LEARNING PIPELINE     ")
    print("=" * 70)

    # 1. Dataset loading / generation
    if not data_path.exists():
        print(f"Dataset not found at {data_path}. Generating synthetic dataset...")
        df = generate_student_data(n_samples=1000, random_seed=42)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Generated 1,000 student records at {data_path}")
    else:
        print(f"Loading existing dataset from {data_path}...")
        df = load_data(data_path)

    # 2. Validation
    validation_summary = validate_data(df)
    print(f"\n[Data Summary] Total Rows: {validation_summary['total_rows']}, Columns: {validation_summary['total_columns']}")
    print(f"[Missing Values Check]:")
    for col, count in validation_summary['missing_counts'].items():
        if count > 0:
            print(f"  - {col}: {count} missing ({validation_summary['missing_pct'][col]}%)")
    if sum(validation_summary['missing_counts'].values()) == 0:
        print("  - No missing values detected.")

    # 3. Train/Val/Test Split (Strict featurization ordering)
    print("\n[Splitting Data] Strict featurization ordering: splitting before transformation...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, test_size=0.15, val_size=0.15, random_state=42
    )
    print(f"  Training set:   {len(X_train)} samples ({len(X_train)/len(df):.0%})")
    print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(df):.0%})")
    print(f"  Test set:       {len(X_test)} samples ({len(X_test)/len(df):.0%})")

    # 4. Model Benchmarking and Cross-Validation
    candidate_models = get_candidate_models(random_state=42)
    benchmark_records = []
    fitted_pipelines = {}

    print("\n[Cross-Validation] 5-Fold Cross-Validation on Training Data:")
    for name, model in candidate_models.items():
        cv_res = perform_cross_validation(model, X_train, y_train, cv=5, random_state=42)
        print(f"  {name:20s} | CV RMSE: {cv_res['cv_rmse_mean']:.3f} (+/- {cv_res['cv_rmse_std']:.3f}) | CV R²: {cv_res['cv_r2_mean']:.3f}")

        # Fit full pipeline on training set
        pipeline = create_model_pipeline(model)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Evaluate on validation set
        y_val_pred = pipeline.predict(X_val)
        val_metrics = compute_metrics(y_val, y_val_pred)

        benchmark_records.append({
            'Model': name,
            'CV_RMSE': cv_res['cv_rmse_mean'],
            'Val_MAE': val_metrics['MAE'],
            'Val_RMSE': val_metrics['RMSE'],
            'Val_R2': val_metrics['R2'],
            'RMSE': val_metrics['RMSE'],  # for plotting helper
            'R2': val_metrics['R2']
        })

    benchmark_df = pd.DataFrame(benchmark_records)
    print("\n[Validation Benchmark Results]:")
    print(benchmark_df[['Model', 'Val_MAE', 'Val_RMSE', 'Val_R2']].to_string(index=False))

    # 5. Champion Selection
    best_record = benchmark_df.sort_values(by='Val_RMSE', ascending=True).iloc[0]
    best_model_name = best_record['Model']
    champion_pipeline = fitted_pipelines[best_model_name]
    print(f"\n Champion Model Selected: {best_model_name} (Lowest Validation RMSE: {best_record['Val_RMSE']:.3f})")

    # 6. Final Evaluation on Holdout Test Set
    print("\n[Holdout Test Set Evaluation]")
    y_test_pred = champion_pipeline.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred)
    print(f"  Model:                 {best_model_name}")
    print(f"  Test MAE:              {test_metrics['MAE']:.3f}")
    print(f"  Test RMSE:             {test_metrics['RMSE']:.3f}")
    print(f"  Test R² Score:         {test_metrics['R2']:.4f}")
    print(f"  Explained Variance:    {test_metrics['Explained_Variance']:.4f}")

    # 7. Save Artifacts & Figures
    model_save_path = models_dir / 'best_model.joblib'
    joblib.dump(champion_pipeline, model_save_path)
    print(f"\n[Saved Model] Best pipeline serialized to: {model_save_path}")

    comparison_fig = figures_dir / 'model_comparison.png'
    plot_model_comparison(benchmark_df, comparison_fig)

    pred_fig = figures_dir / 'actual_vs_predicted.png'
    plot_actual_vs_predicted(y_test, y_test_pred, best_model_name, pred_fig)

    residual_fig = figures_dir / 'residual_plot.png'
    plot_residuals(y_test, y_test_pred, best_model_name, residual_fig)

    print(f"[Generated Visualizations]:")
    print(f"  - {comparison_fig}")
    print(f"  - {pred_fig}")
    print(f"  - {residual_fig}")

    print("\n" + "=" * 70)
    print("                    TRAINING RUN COMPLETE!                         ")
    print("=" * 70)

    return champion_pipeline, test_metrics


if __name__ == '__main__':
    run_pipeline()
