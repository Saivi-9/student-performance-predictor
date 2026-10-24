"""
Explainability and Model Interpretability Module
------------------------------------------------
Provides tools to inspect and explain model predictions:
- Permutation Feature Importance for global feature ranking
- Partial Dependence & Single-Student What-If sensitivity profiling
- Publication-grade importance visualization
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def compute_permutation_importance(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 10,
    scoring: str = "r2",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Computes permutation feature importance on validation data.

    Shuffles each feature column and measures the drop in validation score.
    Higher drop indicates higher reliance on that feature.

    Args:
        pipeline: Fitted end-to-end scikit-learn Pipeline.
        X_val: Validation features DataFrame.
        y_val: Ground truth validation target Series.
        n_repeats: Number of permutations per feature column.
        scoring: Scikit-learn scoring string ('r2', 'neg_root_mean_squared_error', etc.).
        random_state: Seed for reproducibility.

    Returns:
        pd.DataFrame: Table with 'feature', 'importance_mean', 'importance_std' sorted descending.
    """
    result = permutation_importance(
        pipeline,
        X_val,
        y_val,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1
    )

    df_importance = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    return df_importance


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    top_n: int = 10,
    title: str = "Permutation Feature Importance (Validation Drop in R²)"
) -> plt.Figure:
    """
    Generates a horizontal bar chart displaying feature importances with standard deviation error bars.

    Args:
        importance_df: DataFrame containing 'feature', 'importance_mean', 'importance_std'.
        output_path: Optional file path to save the chart image.
        top_n: Number of top features to display.
        title: Title of the chart.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    df_top = importance_df.head(top_n).iloc[::-1]  # Reverse for ascending display in horizontal bar

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(
        df_top["feature"],
        df_top["importance_mean"],
        xerr=df_top["importance_std"],
        color="#3b82f6",
        edgecolor="#1d4ed8",
        capsize=4,
        alpha=0.85
    )

    ax.set_xlabel("Mean Score Decrease (Importance)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    # Annotate numeric values next to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            max(width + 0.005, 0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#1e293b",
            fontweight="bold"
        )

    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300)
        plt.close(fig)

    return fig


def simulate_what_if_trajectory(
    pipeline: Pipeline,
    base_student: Dict[str, Any],
    feature_name: str,
    feature_values: List[Any]
) -> pd.DataFrame:
    """
    Simulates predictions when systematically varying one feature while holding all others constant.

    Args:
        pipeline: Fitted model pipeline.
        base_student: Dictionary representing baseline student attributes.
        feature_name: Attribute name to perturb.
        feature_values: Sequence of candidate values to test.

    Returns:
        pd.DataFrame: Simulation results containing feature value and predicted final score.
    """
    records = []
    for val in feature_values:
        student_copy = base_student.copy()
        student_copy[feature_name] = val
        df_row = pd.DataFrame([student_copy])
        pred = float(pipeline.predict(df_row)[0])
        records.append({
            feature_name: val,
            "predicted_score": round(np.clip(pred, 0.0, 100.0), 2)
        })

    return pd.DataFrame(records)
