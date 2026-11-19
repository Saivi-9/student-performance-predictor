"""
External Open-Source Dataset Fetcher (Hugging Face Hub & UCI Mirrors).

Downloads and caches real benchmark student performance datasets:
1. UCI Student Performance Dataset (DElmazi/Student_Performance: Math & Portuguese, 1,044 rows, 33 cols)
2. 10K Student Performance Dataset (imranx654/Student-Performance-Predictor, 10,000 rows, 6 cols)
"""

import os
from pathlib import Path
from typing import Dict
import pandas as pd
from huggingface_hub import hf_hub_download

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def fetch_uci_dataset(output_dir: Path | str) -> pd.DataFrame:
    """Download and combine real UCI Student Performance Math & Portuguese datasets."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "uci_student_performance.csv"

    if out_file.exists():
        return pd.read_csv(out_file)

    p_mat = hf_hub_download("DElmazi/Student_Performance", "student-mat.csv", repo_type="dataset")
    p_por = hf_hub_download("DElmazi/Student_Performance", "student-por.csv", repo_type="dataset")

    df_mat = pd.read_csv(p_mat, sep=None, engine="python")
    df_mat["subject"] = "Math"
    df_por = pd.read_csv(p_por, sep=None, engine="python")
    df_por["subject"] = "Portuguese"

    df_uci = pd.concat([df_mat, df_por], ignore_index=True)
    df_uci.to_csv(out_file, index=False)
    return df_uci


def fetch_hf_10k_dataset(output_dir: Path | str) -> pd.DataFrame:
    """Download the 10,000-row StudentPerformance benchmark dataset from Hugging Face."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hf_student_performance_10k.csv"

    if out_file.exists():
        return pd.read_csv(out_file)

    p_csv = hf_hub_download("imranx654/Student-Performance-Predictor", "Student_performace/StudentPerformance.csv")
    df_10k = pd.read_csv(p_csv)
    df_10k.to_csv(out_file, index=False)
    return df_10k


def fetch_all_external_datasets(base_dir: Path | str = None) -> Dict[str, pd.DataFrame]:
    """Fetch and cache all external open-source benchmark datasets."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent / "external"
    else:
        base_dir = Path(base_dir)

    datasets = {}
    try:
        datasets["uci"] = fetch_uci_dataset(base_dir)
    except Exception as exc:
        print(f"[Warning] Could not fetch UCI dataset: {exc}")

    try:
        datasets["hf_10k"] = fetch_hf_10k_dataset(base_dir)
    except Exception as exc:
        print(f"[Warning] Could not fetch HF 10K dataset: {exc}")

    return datasets


if __name__ == "__main__":
    res = fetch_all_external_datasets()
    for name, df in res.items():
        print(f"Fetched '{name}': {df.shape[0]} rows x {df.shape[1]} columns")
