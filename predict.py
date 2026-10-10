"""
Inference script for predicting student final exam score.
Supports command-line arguments and an interactive questionnaire mode.
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd


def get_performance_tier(score: float) -> str:
    """Classify the predicted numeric score into an intuitive grade tier."""
    if score >= 90.0:
        return "Grade A (Excellent - Outstanding Performance)"
    elif score >= 80.0:
        return "Grade B (Very Good - Strong Grasp)"
    elif score >= 70.0:
        return "Grade C (Good - Satisfactory)"
    elif score >= 60.0:
        return "Grade D (Passing - Needs Reinforcement)"
    else:
        return "Grade F (At Risk - Intensive Tutoring Recommended)"


def predict_score(input_data: dict, model_path: Path | str = None) -> dict:
    """Predict student final score from input features dict.

    Args:
        input_data: Dictionary of feature key-values.
        model_path: Optional explicit path to saved pipeline.

    Returns:
        dict: Prediction results including raw score and tier.
    """
    if model_path is None:
        project_root = Path(__file__).resolve().parent
        model_path = project_root / 'saved_models' / 'best_model.joblib'

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python train.py' first to train and save the model."
        )

    pipeline = joblib.load(model_path)
    df = pd.DataFrame([input_data])
    raw_pred = float(pipeline.predict(df)[0])
    bounded_pred = round(max(0.0, min(100.0, raw_pred)), 1)

    return {
        'predicted_score': bounded_pred,
        'performance_tier': get_performance_tier(bounded_pred),
        'input_features': input_data
    }


def interactive_mode():
    """Prompt the user interactively in the terminal."""
    print("\n" + "=" * 60)
    print("      STUDENT PERFORMANCE PREDICTOR - INTERACTIVE CLI       ")
    print("=" * 60)
    print("Please answer the following questions about the student:\n")

    try:
        hours = float(input("1. Hours studied per week (e.g. 1.0 - 15.0) [default 5.5]: ") or 5.5)
        attendance = float(input("2. Attendance rate in % (e.g. 50 - 100) [default 85.0]: ") or 85.0)
        prev_score = float(input("3. Previous exam score (0 - 100) [default 75.0]: ") or 75.0)
        sleep = float(input("4. Average sleep hours per night (4 - 10) [default 7.0]: ") or 7.0)
        tutoring = int(input("5. Number of tutoring sessions attended (0 - 10) [default 2]: ") or 2)

        parental = input("6. Parental involvement (Low / Medium / High) [default Medium]: ").strip().capitalize() or "Medium"
        if parental not in ['Low', 'Medium', 'High']:
            parental = "Medium"

        extra = input("7. Extracurricular activities? (Yes / No) [default Yes]: ").strip().capitalize() or "Yes"
        if extra not in ['Yes', 'No']:
            extra = "Yes"

        internet = input("8. Internet access at home? (Yes / No) [default Yes]: ").strip().capitalize() or "Yes"
        if internet not in ['Yes', 'No']:
            internet = "Yes"

        features = {
            'hours_studied': hours,
            'attendance_percent': attendance,
            'previous_score': prev_score,
            'sleep_hours': sleep,
            'tutoring_sessions': tutoring,
            'parental_involvement': parental,
            'extracurricular': extra,
            'internet_access': internet
        }

        result = predict_score(features)
        print("\n" + "-" * 60)
        print("                 PREDICTION RESULTS                         ")
        print("-" * 60)
        print(f" Predicted Final Exam Score : {result['predicted_score']} / 100")
        print(f" Academic Performance Tier  : {result['performance_tier']}")
        print("-" * 60)

    except Exception as e:
        print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(description="Predict student final exam score using trained ML pipeline.")
    parser.add_argument('--hours', type=float, help="Hours studied per week")
    parser.add_argument('--attendance', type=float, help="Attendance percentage (0-100)")
    parser.add_argument('--prev-score', type=float, help="Previous exam score (0-100)")
    parser.add_argument('--sleep', type=float, default=7.0, help="Average sleep hours per night")
    parser.add_argument('--tutoring', type=int, default=2, help="Tutoring sessions attended")
    parser.add_argument('--parental', type=str, default='Medium', choices=['Low', 'Medium', 'High'], help="Parental involvement level")
    parser.add_argument('--extracurricular', type=str, default='Yes', choices=['Yes', 'No'], help="Extracurricular participation")
    parser.add_argument('--internet', type=str, default='Yes', choices=['Yes', 'No'], help="Internet access at home")

    args = parser.parse_args()

    if args.hours is not None and args.attendance is not None and args.prev_score is not None:
        features = {
            'hours_studied': args.hours,
            'attendance_percent': args.attendance,
            'previous_score': args.prev_score,
            'sleep_hours': args.sleep,
            'tutoring_sessions': args.tutoring,
            'parental_involvement': args.parental,
            'extracurricular': args.extracurricular,
            'internet_access': args.internet
        }
        res = predict_score(features)
        print(f"Predicted Score: {res['predicted_score']} | Tier: {res['performance_tier']}")
    else:
        interactive_mode()


if __name__ == '__main__':
    main()
