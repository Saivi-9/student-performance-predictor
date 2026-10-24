# 🎓 Student Performance Predictor (Machine Learning Project)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-17%2F17%20Passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-structured machine learning project designed for portfolio showcases and educational decision support. This project demonstrates how to ingest data, handle missing values, engineer informative behavioral features, build leak-free scikit-learn preprocessing pipelines, perform $k$-fold cross-validation and hyperparameter grid tuning, interpret model predictions via permutation importance, and deploy an interactive Streamlit web dashboard and CLI inference tool.

---

## 📌 Problem Overview

Can we reliably estimate a student's final examination score based on their study habits, attendance record, and socioeconomic factors?

By identifying key academic drivers and predicting final outcomes, educational institutions and mentors can implement targeted academic interventions before final exams.

---

## 🔬 Key Machine Learning Concepts Demonstrated

1. **Strict Featurization Ordering (No Data Leakage)**:
   - The dataset is split into **Train (70%)**, **Validation (15%)**, and **Test (15%)** sets *before* fitting any scalers, encoders, or imputers.
   - All transformations are computed strictly on the training set and applied downstream.

2. **Feature Engineering Module (`src/features.py`)**:
   - `study_efficiency`: Study hours relative to tutoring session support.
   - `academic_momentum`: Weighted combination of prior benchmark scores and attendance rate.
   - `sleep_deficit`: Absolute deviation from recommended 8-hour sleep window.
   - `engagement_score`: Composite normalized engagement index across attendance, study, and tutoring.

3. **Unified Preprocessing Pipeline (`ColumnTransformer`)**:
   - **Numerical Features**: Imputed using the `median` and scaled via `StandardScaler` (Z-score normalization).
   - **Ordinal Features**: Imputed using `most_frequent` and mapped using `OrdinalEncoder` (`Low` < `Medium` < `High`).
   - **Nominal Categorical Features**: Imputed using `most_frequent` and transformed via `OneHotEncoder(drop='first')` to prevent the dummy variable trap.

4. **5-Fold Cross-Validation & Automated Hyperparameter Grid Tuning (`tune.py`)**:
   - Benchmarks **Linear Regression**, regularized **Ridge Regression** (with optimal $L_2$ alpha search), and non-linear **Random Forest Regressors**.

5. **Model Interpretability & Explainability (`src/explainability.py`)**:
   - Permutation Feature Importance calculating drops in validation $R^2$.
   - Single-student 'What-If' sensitivity analysis profiling intervention trajectories.

6. **Interactive Streamlit Web Dashboard & CLI Inference**:
   - Multi-tab decision support web application (`app.py`) and command-line inference tool (`predict.py`).

---

## 📊 Dataset Schema

The dataset contains 1,000 student records with 8 predictive features and a continuous target:

| Feature Name | Type | Description | Values / Range |
| :--- | :--- | :--- | :--- |
| `hours_studied` | Numerical | Weekly hours dedicated to independent study | `1.0` - `12.0` hrs |
| `attendance_percent` | Numerical | School attendance rate | `50.0%` - `100.0%` |
| `previous_score` | Numerical | Score achieved on previous midterm test | `35.0` - `98.0` / 100 |
| `sleep_hours` | Numerical | Average sleep duration per night | `4.0` - `10.0` hrs |
| `tutoring_sessions` | Numerical | Number of tutoring sessions attended | `0` - `8` sessions |
| `parental_involvement`| Categorical (Ordinal) | Level of parental academic support | `Low`, `Medium`, `High` |
| `extracurricular` | Categorical (Nominal) | Participation in school clubs / sports | `Yes`, `No` |
| `internet_access` | Categorical (Nominal) | High-speed internet access at home | `Yes`, `No` |
| **`final_score`** | **Target (Continuous)** | **Final examination outcome** | **`0.0` - `100.0` pts** |

---

## 🏆 Model Benchmark & Performance

Evaluation results on the validation and unseen test sets:

| Model | 5-Fold CV RMSE | Val MAE | Val RMSE | Val R² Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression (Champion)** | **3.203** | **2.442** | **3.146** | **0.8481** |
| **Ridge Regression** | 3.202 | 2.444 | 3.148 | 0.8479 |
| **Random Forest Regressor** | 4.084 | 3.357 | 4.275 | 0.7195 |

### Final Evaluation on Unseen Test Set (Champion Model)
- **Mean Absolute Error (MAE)**: `2.366` points
- **Root Mean Squared Error (RMSE)**: `2.984` points
- **R² Score**: `0.8385` (explains **~83.9%** of score variance)
- **Explained Variance**: `0.8388`

Generated evaluation charts are stored in `reports/figures/`:
- `model_comparison.png`: Side-by-side benchmark bar chart.
- `actual_vs_predicted.png`: Scatter plot with diagonal perfect-fit line ($y = x$).
- `residual_plot.png`: Homoscedasticity verification checking residual distribution.

---

## 📁 Repository Structure

```text
student-performance-predictor/
├── data/
│   ├── generate_dataset.py           # Reproducible synthetic dataset generator
│   └── student_data.csv              # Student tabular dataset (1,000 samples)
├── notebooks/
│   └── student_performance_eda.ipynb # Narrative-driven exploratory analysis
├── reports/
│   └── figures/                      # Exported high-res diagnostic plots
│       ├── actual_vs_predicted.png
│       ├── model_comparison.png
│       └── residual_plot.png
├── saved_models/
│   └── best_model.joblib             # Serialized champion scikit-learn pipeline
├── src/
│   ├── __init__.py
│   ├── dataset.py                    # Ingestion, validation, and leak-free splitting
│   ├── evaluate.py                   # Regression metrics and plotting utilities
│   ├── explainability.py             # Permutation feature importance & What-If trajectories
│   ├── features.py                   # Engineered domain features & composite scores
│   ├── models.py                     # Estimator definitions & cross-validation
│   ├── pipeline.py                   # ColumnTransformer and scikit-learn Pipelines
│   └── tuning.py                     # Automated GridSearchCV hyperparameter optimization
├── tests/
│   ├── test_explainability.py        # Tests for importance & sensitivity simulations
│   ├── test_features.py              # Tests for engineered features & missing value resilience
│   ├── test_pipeline.py              # Tests for leak-free transformations & candidates
│   └── test_tuning.py                # Tests for Ridge & Random Forest parameter tuning
├── app.py                            # Interactive multi-tab Streamlit web application
├── predict.py                        # Real-time CLI inference script
├── tune.py                           # CLI for automated hyperparameter optimization
├── train.py                          # Master training orchestration script
├── pytest.ini                        # Pytest test configuration
├── requirements.txt                  # Project package dependencies
└── README.md                         # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Saivi-9/student-performance-predictor.git
cd student-performance-predictor
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch Interactive Streamlit Web Application
Launch the multi-tab decision support dashboard:
```bash
streamlit run app.py
```
**Dashboard Features:**
- 🎯 **Live Score Forecasting:** Real-time exam score projection with institutional grade badges (Grade A to F) and study density metrics.
- 🔮 **'What-If' Intervention Studio:** Interactive sensitivity simulator testing how incremental changes in study hours, attendance, or tutoring alter the student's projected grade tier.
- 📊 **Explainability & Transparency:** Global permutation feature importance showing empirical score drops when features are disrupted.
- 📈 **Benchmark & Model Diagnostics:** Review test set actual-vs-predicted correlations and error homoscedasticity.

### 4. Automated Hyperparameter Grid Tuning CLI
Execute systematic grid search cross-validation to optimize model hyperparameters:
```bash
# Tune Ridge regression L2 penalty with 5-fold CV and save the champion pipeline
python tune.py --model ridge --cv 5 --save

# Tune Random Forest tree depth and split criteria with 5-fold CV
python tune.py --model rf --cv 5
```

### 5. Run the Master Training Pipeline
Trains candidate models, executes 5-fold cross-validation, benchmarks test set performance, and exports diagnostic plots to `reports/figures/`:
```bash
python train.py
```

### 6. Run CLI Predictions

**Interactive Questionnaire Mode:**
```bash
python predict.py
```

**Single Command Flag Mode:**
```bash
python predict.py --hours 7.5 --attendance 92.0 --prev-score 85.0 --sleep 8.0 --tutoring 3 --parental High --extracurricular Yes --internet Yes
```
Output:
```text
Predicted Score: 94.2 | Tier: Grade A (Excellent - Outstanding Performance)
```

### 7. Run Full Unit Test Suite
Execute all 17 passing unit tests across feature engineering, leak-free pipelines, explainability, and tuning:
```bash
pytest tests/ -v
```
Output:
```text
============================= 17 passed in 14.46s =============================
```

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
