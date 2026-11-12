# 🎓 Student Academic Performance Predictor Studio (v2.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9%2B-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7-success.svg)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19%20%2B%20Vite-61DAFB.svg)](https://react.dev/)
[![Hugging Face Hub](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Pretrained%20Hub-yellow.svg)](https://huggingface.co/)
[![Tests](https://img.shields.io/badge/Tests-28%2F28%20Passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end machine learning engineering platform and real-time analytics studio built for student performance forecasting, academic intervention simulation, and Hugging Face pre-trained model benchmarking.

---

## ✨ What's New in v2.0

1. **20x More Training Data & Multi-Source Benchmarks (`data/`)**:
   - **Primary Cohort (`data/student_data.csv`)**: **20,000 student records** across **20 raw features** and non-linear educational/biological dynamics with realistic ~1.0–1.8% missingness.
   - **Real Open Benchmarks (`data/fetch_external.py`)**: Automatically fetches & caches Hugging Face's **UCI Student Performance (Math + Portuguese)** (`1,044` rows × `34` cols via `DElmazi/Student_Performance`) and **10K Student Performance Index** (`10,000` rows × `6` cols via `imranx654/Student-Performance-Predictor`).

2. **28 Total Pipeline Features (`src/features.py`, `src/pipeline.py`)**:
   - **20 Raw Features**:
     - *Academic (6)*: `previous_score`, `quiz_average`, `hours_studied`, `attendance_percent`, `assignment_completion_rate`, `practice_papers_solved`
     - *Wellbeing & Habits (6)*: `sleep_hours`, `tutoring_sessions`, `physical_activity_hours`, `screen_time_hours`, `stress_level`, `commute_minutes`
     - *Support & Context (8)*: `parental_involvement`, `motivation_level`, `access_to_resources`, `parental_education`, `extracurricular`, `internet_access`, `school_type`, `learning_style`
   - **8 Leak-Free Engineered Composite Features** (computed inside `AcademicFeatureEngineer` fitted strictly on `X_train`):
     - `study_efficiency`, `academic_momentum`, `sleep_deficit`, `engagement_score`, `study_sleep_balance`, `digital_distraction_ratio`, `formative_delta`, `wellbeing_index`

3. **9-Model Cross-Validation Tournament (`src/models.py`, `train.py`)**:
   - Benchmarks **MLP Neural Network**, **Stacking Ensemble**, **LightGBM**, **HistGradientBoosting**, **Ridge Regression**, **Linear Regression**, **ElasticNet**, **Extra Trees**, and **Random Forest** using 5-fold CV and strict 70/15/15 Train/Val/Test splits.
   - All 9 trained model pipelines are saved to `saved_models/registry/` for real-time switching in the UI and REST API.

4. **Hugging Face Pre-Trained Model & Dataset Hub (`src/pretrained_hub.py`)**:
   - Live keyword search across Hugging Face models and datasets (`search_huggingface_models`, `search_huggingface_datasets`).
   - Curated catalog + real benchmark evaluation & live inference on community pre-trained models:
     - `Shalan1/student-performance-predictor-models` (Random Forest Regressor evaluated on 10K dataset: **$R^2 = 0.9952$**, RMSE `1.330`)
     - `SergeyR256/student-performance` (37-feature UCI Random Forest Pipeline evaluated on 649 Portuguese students: **$R^2 = 0.6454$**, RMSE `1.922`)

5. **Real-Time Interactive Studio UI (`frontend/` + `src/api/` + `app.py`)**:
   - **FastAPI + React/Vite Studio (`api_server.py`)**: Live 20-slider forecaster with instant confidence intervals & local factor attributions, live engine switcher across all 9 models, 9-lever What-If Sensitivity Curve simulator, real benchmark & residual scatter explorer, HF Pre-Trained Hub search/comparison, and batch cohort scoring.
   - **Streamlit Multi-Tab Dashboard (`app.py`)**: Interactive 5-tab analytics & batch CSV scoring workflow.

---

## 🏆 9-Model Benchmark Tournament Results (20,000 Students, 28 Features)

| Architecture | 5-Fold CV RMSE | Val RMSE | Val $R^2$ | Test RMSE | Test MAE | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP Neural Network (Champion)** | **2.489** | **2.429** | **0.9716** | **2.508** | **2.008** | **0.9688** |
| **Stacking Ensemble** | 2.532 | 2.477 | 0.9704 | 2.544 | 2.020 | 0.9679 |
| **LightGBM** | 2.561 | 2.515 | 0.9695 | 2.561 | 2.021 | 0.9675 |
| **HistGradientBoosting** | 2.588 | 2.544 | 0.9688 | 2.617 | 2.059 | 0.9660 |
| **Ridge Regression** | 2.997 | 2.892 | 0.9597 | 2.941 | 2.265 | 0.9571 |
| **Linear Regression** | 2.997 | 2.892 | 0.9597 | 2.941 | 2.265 | 0.9571 |
| **ElasticNet** | 3.019 | 2.921 | 0.9589 | 2.963 | 2.293 | 0.9564 |
| **Extra Trees** | 3.376 | 3.323 | 0.9468 | 3.375 | 2.639 | 0.9435 |
| **Random Forest** | 3.493 | 3.442 | 0.9429 | 3.473 | 2.701 | 0.9402 |

Diagnostic figures exported automatically by `train.py` into `reports/figures/`:
- `model_comparison.png`, `actual_vs_predicted.png`, `residual_plot.png`, `feature_importance.png`
- Full telemetry saved to `reports/benchmark_results.json`

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
git clone https://github.com/Saivi-9/student-performance-predictor.git
cd student-performance-predictor

python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Full 9-Model Training Pipeline (Optional — Pre-Trained Weights Included)
Generates/loads the 20k dataset, caches external UCI & HF 10K datasets, benchmarks all 9 models via 5-fold CV, and saves `saved_models/best_model.joblib` + `saved_models/registry/*.joblib`:
```bash
python train.py
```

### 3. Launch the Real-Time Web Studio (FastAPI + Built React App)
```bash
python api_server.py --port 8001
```
Then open **`http://localhost:8001`** in your browser for the full real-time studio, or **`http://localhost:8001/docs`** for interactive OpenAPI/Swagger documentation.

To launch the Streamlit dashboard alternatively:
```bash
streamlit run app.py
```

### 4. CLI Hyperparameter Tuning & Inference
```bash
# Tune LightGBM, Ridge, or Random Forest with k-fold CV
python tune.py --model lgbm --cv 5 --save

# Run single-student CLI prediction using any registered model
python predict.py --hours 14 --attendance 92 --prev-score 82 --model-slug lightgbm
```

### 5. Run Full Unit & API Test Suite
```bash
pytest tests/ -v
```

---

## 📁 Repository Structure

```text
student-performance-predictor/
├── api_server.py                     # Uvicorn entrypoint serving FastAPI + React/Vite UI
├── app.py                            # 5-tab Streamlit Analytics Dashboard
├── train.py                          # 9-model CV tournament, HF dataset cache & reporting
├── tune.py                           # Hyperparameter tuning CLI (Ridge, RF, LightGBM)
├── predict.py                        # 20-feature CLI predictor with --model-slug switching
├── data/
│   ├── generate_dataset.py           # 20,000-row, 20-feature non-linear dataset generator
│   ├── fetch_external.py             # UCI & HF 10K real open dataset downloader/cacher
│   └── student_data.csv              # Primary cohort dataset
├── frontend/                         # React 19 + TypeScript + Vite + Tailwind Studio UI
│   └── src/components/
│       ├── StudentForecaster.tsx     # 20-feature real-time predictor & factor attributions
│       ├── SimulationStudio.tsx      # 9-lever sensitivity curve simulator
│       ├── ModelBenchmarks.tsx       # Live tournament leaderboard & scatter diagnostics
│       └── PretrainedHub.tsx         # HF Hub search, benchmark, cohort & batch scorer
├── src/
│   ├── dataset.py                    # Schema validation & leak-free 70/15/15 splitter
│   ├── features.py                   # 8 composite features + AcademicFeatureEngineer
│   ├── pipeline.py                   # End-to-end ColumnTransformer + estimator pipelines
│   ├── models.py                     # 9 model architectures & CV tournament runner
│   ├── pretrained_hub.py             # Hugging Face Hub search, catalog & benchmarking
│   ├── tuning.py                     # GridSearchCV / RandomizedSearchCV utilities
│   ├── explainability.py             # Permutation importance & What-If trajectories
│   └── api/                          # FastAPI routes & Pydantic v2 schemas
├── tests/                            # 28 unit & integration tests across ML & REST API
└── reports/                          # Benchmark JSON telemetry & diagnostic charts
```

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
