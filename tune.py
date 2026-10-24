"""
Automated Hyperparameter Tuning CLI Script.
Executes systematic grid search cross-validation across regression model hyperparameter spaces.
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd
from src.dataset import load_data, split_data
from src.tuning import tune_ridge_pipeline, tune_random_forest_pipeline
from src.evaluate import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Automated hyperparameter optimization with k-fold CV.")
    parser.add_argument(
        "--model",
        type=str,
        default="ridge",
        choices=["ridge", "rf"],
        help="Model architecture to tune (ridge or rf)"
    )
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--save", action="store_true", help="Save the tuned model to saved_models/tuned_model.joblib")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "student_data.csv"
    models_dir = project_root / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      STUDENT PERFORMANCE PREDICTOR - HYPERPARAMETER TUNING         ")
    print("=" * 70)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run train.py first.")

    df = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, test_size=0.15, val_size=0.15, random_state=42
    )
    print(f"Loaded dataset: {len(df)} total rows. Training split: {len(X_train)} samples.")

    if args.model == "ridge":
        print(f"\n[Running Ridge Grid Tuning] {args.cv}-fold CV across alpha values...")
        alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
        best_pipeline, best_params, results_df = tune_ridge_pipeline(
            X_train, y_train, alphas=alphas, cv=args.cv, random_state=42
        )
        print("\nTop 5 Alpha Candidate Results (Lowest CV RMSE):")
        print(results_df.head(5).to_string(index=False))

    else:
        print(f"\n[Running Random Forest Grid Tuning] {args.cv}-fold CV...")
        param_grid = {
            "model__n_estimators": [60, 100, 140],
            "model__max_depth": [5, 8, 12],
            "model__min_samples_split": [2, 4],
        }
        best_pipeline, best_params, results_df = tune_random_forest_pipeline(
            X_train, y_train, param_grid=param_grid, cv=args.cv, random_state=42
        )
        print("\nTop 5 Parameter Configurations:")
        print(results_df.head(5).to_string(index=False))

    print("\n" + "-" * 70)
    print(f"🎯 Optimal Parameters: {best_params}")

    # Evaluate best tuned pipeline on validation split
    y_val_pred = best_pipeline.predict(X_val)
    val_metrics = compute_metrics(y_val, y_val_pred)
    print(f"Validation Performance: RMSE = {val_metrics['RMSE']:.3f} | R² = {val_metrics['R2']:.4f} | MAE = {val_metrics['MAE']:.3f}")
    print("-" * 70)

    if args.save:
        save_path = models_dir / "tuned_model.joblib"
        joblib.dump(best_pipeline, save_path)
        print(f"✅ Tuned pipeline successfully saved to {save_path}")


if __name__ == "__main__":
    main()
