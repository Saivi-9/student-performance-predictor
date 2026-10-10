"""
Evaluation and visualization module for regression models.
Computes metrics (MAE, RMSE, R²) and generates diagnostic plots.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate key regression evaluation metrics.

    Args:
        y_true: True ground truth values.
        y_pred: Predicted values from model.

    Returns:
        Dict[str, float]: Dictionary of metric names and values.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    return {
        'MAE': round(float(mae), 3),
        'MSE': round(float(mse), 3),
        'RMSE': round(float(rmse), 3),
        'R2': round(float(r2), 4),
        'Explained_Variance': round(float(evs), 4)
    }


def plot_actual_vs_predicted(y_true, y_pred, model_name: str, output_path: str | Path):
    """Plot actual vs predicted scatter plot with the ideal reference diagonal line.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        model_name: Name of model for title.
        output_path: Path to save figure.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='#2563eb', edgecolors='k', linewidth=0.5, label='Students')

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', linewidth=2, label='Perfect Fit (y = x)')

    plt.xlabel('Actual Final Score', fontsize=11, fontweight='bold')
    plt.ylabel('Predicted Final Score', fontsize=11, fontweight='bold')
    plt.title(f'Actual vs. Predicted Scores - {model_name}', fontsize=12, fontweight='bold', pad=12)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_residuals(y_true, y_pred, model_name: str, output_path: str | Path):
    """Plot residuals (y_true - y_pred) against predicted values to verify error homoscedasticity.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        model_name: Name of model for title.
        output_path: Path to save figure.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    residuals = np.array(y_true) - np.array(y_pred)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color='#059669', edgecolors='k', linewidth=0.5)
    plt.axhline(0, color='#dc2626', linestyle='--', linewidth=2, label='Zero Error')

    plt.xlabel('Predicted Final Score', fontsize=11, fontweight='bold')
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=11, fontweight='bold')
    plt.title(f'Residual Analysis - {model_name}', fontsize=12, fontweight='bold', pad=12)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_model_comparison(results_df, output_path: str | Path):
    """Plot a comparison bar chart across candidate models for RMSE and R2.

    Args:
        results_df: DataFrame with 'Model', 'RMSE', 'R2' columns.
        output_path: Path to save figure.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(results_df))
    width = 0.35

    color1 = '#3b82f6'
    color2 = '#10b981'

    bars1 = ax1.bar(x - width/2, results_df['RMSE'], width, label='RMSE (Lower is better)', color=color1)
    ax1.set_ylabel('RMSE', color=color1, fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['Model'], fontsize=10, fontweight='bold')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, results_df['R2'], width, label='R² Score (Higher is better)', color=color2)
    ax2.set_ylabel('R² Score', color=color2, fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0, 1.05])

    plt.title('Benchmark Comparison Across Machine Learning Models', fontsize=12, fontweight='bold', pad=14)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
