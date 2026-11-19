"""
Inference script for predicting student final exam score (v2.0).
Supports all 20 raw features with smart defaults, model registry switching via --model-slug,
and an interactive CLI questionnaire mode.
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd
from src.dataset import DEFAULT_STUDENT_PROFILE
from src.features import engineer_features


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


def predict_score(
    input_data: dict,
    model_path: Path | str = None,
    model_slug: str | None = None,
) -> dict:
    """Predict student final score from input features dict (auto-completing defaults)."""
    project_root = Path(__file__).resolve().parent
    if model_path is None:
        if model_slug:
            model_path = project_root / 'saved_models' / 'registry' / f'{model_slug}.joblib'
        else:
            model_path = project_root / 'saved_models' / 'best_model.joblib'

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python train.py' first to train and save the models."
        )

    complete_data = {**DEFAULT_STUDENT_PROFILE, **input_data}
    pipeline = joblib.load(model_path)
    df = pd.DataFrame([complete_data])
    raw_pred = float(pipeline.predict(df)[0])
    bounded_pred = round(max(0.0, min(100.0, raw_pred)), 1)

    eng = engineer_features(df)
    engineered_summary = {
        "study_efficiency": round(float(eng["study_efficiency"].iloc[0]), 3),
        "academic_momentum": round(float(eng["academic_momentum"].iloc[0]), 2),
        "engagement_score": round(float(eng["engagement_score"].iloc[0]), 3),
        "wellbeing_index": round(float(eng["wellbeing_index"].iloc[0]), 3),
    }

    return {
        'predicted_score': bounded_pred,
        'performance_tier': get_performance_tier(bounded_pred),
        'model_used': model_slug or 'best_model',
        'engineered_features': engineered_summary,
        'input_features': complete_data,
    }


def interactive_mode(model_slug: str | None = None):
    """Prompt the user interactively in the terminal."""
    print("\n" + "=" * 65)
    print("   STUDENT PERFORMANCE PREDICTOR v2.0 - INTERACTIVE CLI   ")
    print("=" * 65)
    print("Press Enter to accept any default value.\n")

    try:
        hours = float(input("1. Weekly study hours (1-40) [12.0]: ") or 12.0)
        attendance = float(input("2. Attendance rate % (40-100) [88.0]: ") or 88.0)
        prev_score = float(input("3. Previous exam score (0-100) [74.0]: ") or 74.0)
        quiz_avg = float(input("4. Formative quiz average (0-100) [78.0]: ") or 78.0)
        assign_rate = float(input("5. Assignment completion % (0-100) [85.0]: ") or 85.0)
        papers = float(input("6. Practice papers solved (0-20) [5]: ") or 5)
        sleep = float(input("7. Sleep hours/night (4-12) [7.5]: ") or 7.5)
        tutoring = int(input("8. Tutoring sessions/month (0-10) [2]: ") or 2)
        phys = float(input("9. Physical activity hrs/week (0-20) [4.5]: ") or 4.5)
        screen = float(input("10. Recreational screen time hrs/day (0-12) [3.0]: ") or 3.0)
        stress = float(input("11. Stress level (1-10) [4]: ") or 4)
        commute = float(input("12. Daily commute minutes (0-120) [25]: ") or 25)

        parental = input("13. Parental involvement (Low/Medium/High) [Medium]: ").strip().capitalize() or "Medium"
        motivation = input("14. Motivation level (Low/Medium/High) [Medium]: ").strip().capitalize() or "Medium"
        resources = input("15. Access to resources (Low/Medium/High) [Medium]: ").strip().capitalize() or "Medium"
        parent_edu = input("16. Parental education (High School/Bachelor/Master+) [Bachelor]: ").strip() or "Bachelor"
        extra = input("17. Extracurricular activities (Yes/No) [Yes]: ").strip().capitalize() or "Yes"
        internet = input("18. Internet access at home (Yes/No) [Yes]: ").strip().capitalize() or "Yes"
        school_type = input("19. School type (Public/Private) [Public]: ").strip().capitalize() or "Public"
        learning_style = input("20. Learning style (Visual/Auditory/Kinesthetic/Reading/Writing) [Visual]: ").strip() or "Visual"

        features = {
            'hours_studied': hours,
            'attendance_percent': attendance,
            'previous_score': prev_score,
            'quiz_average': quiz_avg,
            'assignment_completion_rate': assign_rate,
            'practice_papers_solved': papers,
            'sleep_hours': sleep,
            'tutoring_sessions': tutoring,
            'physical_activity_hours': phys,
            'screen_time_hours': screen,
            'stress_level': stress,
            'commute_minutes': commute,
            'parental_involvement': parental if parental in ['Low', 'Medium', 'High'] else 'Medium',
            'motivation_level': motivation if motivation in ['Low', 'Medium', 'High'] else 'Medium',
            'access_to_resources': resources if resources in ['Low', 'Medium', 'High'] else 'Medium',
            'parental_education': parent_edu if parent_edu in ['High School', 'Bachelor', 'Master+'] else 'Bachelor',
            'extracurricular': extra if extra in ['Yes', 'No'] else 'Yes',
            'internet_access': internet if internet in ['Yes', 'No'] else 'Yes',
            'school_type': school_type if school_type in ['Public', 'Private'] else 'Public',
            'learning_style': learning_style if learning_style in ['Visual', 'Auditory', 'Kinesthetic', 'Reading/Writing'] else 'Visual',
        }

        result = predict_score(features, model_slug=model_slug)
        print("\n" + "-" * 65)
        print("                 PREDICTION RESULTS (v2.0)                   ")
        print("-" * 65)
        print(f" Model Used                 : {result['model_used']}")
        print(f" Predicted Final Exam Score : {result['predicted_score']} / 100")
        print(f" Academic Performance Tier  : {result['performance_tier']}")
        print(f" Engineered Metrics         : {result['engineered_features']}")
        print("-" * 65)

    except Exception as e:
        print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(description="Predict student final exam score using trained ML pipeline (v2.0).")
    parser.add_argument('--hours', type=float, help="Hours studied per week")
    parser.add_argument('--attendance', type=float, help="Attendance percentage (0-100)")
    parser.add_argument('--prev-score', type=float, help="Previous exam score (0-100)")
    parser.add_argument('--quiz-avg', type=float, default=DEFAULT_STUDENT_PROFILE['quiz_average'], help="Quiz average (0-100)")
    parser.add_argument('--assignments', type=float, default=DEFAULT_STUDENT_PROFILE['assignment_completion_rate'], help="Assignment completion percentage (0-100)")
    parser.add_argument('--papers', type=float, default=DEFAULT_STUDENT_PROFILE['practice_papers_solved'], help="Practice papers solved")
    parser.add_argument('--sleep', type=float, default=DEFAULT_STUDENT_PROFILE['sleep_hours'], help="Average sleep hours per night")
    parser.add_argument('--tutoring', type=int, default=DEFAULT_STUDENT_PROFILE['tutoring_sessions'], help="Tutoring sessions attended")
    parser.add_argument('--physical', type=float, default=DEFAULT_STUDENT_PROFILE['physical_activity_hours'], help="Physical activity hrs/week")
    parser.add_argument('--screen-time', type=float, default=DEFAULT_STUDENT_PROFILE['screen_time_hours'], help="Screen time hrs/day")
    parser.add_argument('--stress', type=float, default=DEFAULT_STUDENT_PROFILE['stress_level'], help="Stress level (1-10)")
    parser.add_argument('--commute', type=float, default=DEFAULT_STUDENT_PROFILE['commute_minutes'], help="Commute minutes/day")
    parser.add_argument('--parental', type=str, default='Medium', choices=['Low', 'Medium', 'High'])
    parser.add_argument('--motivation', type=str, default='Medium', choices=['Low', 'Medium', 'High'])
    parser.add_argument('--resources', type=str, default='Medium', choices=['Low', 'Medium', 'High'])
    parser.add_argument('--parent-edu', type=str, default='Bachelor', choices=['High School', 'Bachelor', 'Master+'])
    parser.add_argument('--extracurricular', type=str, default='Yes', choices=['Yes', 'No'])
    parser.add_argument('--internet', type=str, default='Yes', choices=['Yes', 'No'])
    parser.add_argument('--school-type', type=str, default='Public', choices=['Public', 'Private'])
    parser.add_argument('--learning-style', type=str, default='Visual', choices=['Visual', 'Auditory', 'Kinesthetic', 'Reading/Writing'])
    parser.add_argument('--model-slug', type=str, default=None, help="Optional model slug from saved_models/registry/")

    args = parser.parse_args()

    if args.hours is not None and args.attendance is not None and args.prev_score is not None:
        features = {
            'hours_studied': args.hours,
            'attendance_percent': args.attendance,
            'previous_score': args.prev_score,
            'quiz_average': args.quiz_avg,
            'assignment_completion_rate': args.assignments,
            'practice_papers_solved': args.papers,
            'sleep_hours': args.sleep,
            'tutoring_sessions': args.tutoring,
            'physical_activity_hours': args.physical,
            'screen_time_hours': args.screen_time,
            'stress_level': args.stress,
            'commute_minutes': args.commute,
            'parental_involvement': args.parental,
            'motivation_level': args.motivation,
            'access_to_resources': args.resources,
            'parental_education': args.parent_edu,
            'extracurricular': args.extracurricular,
            'internet_access': args.internet,
            'school_type': args.school_type,
            'learning_style': args.learning_style,
        }
        res = predict_score(features, model_slug=args.model_slug)
        print(f"Model: {res['model_used']} | Predicted Score: {res['predicted_score']} | Tier: {res['performance_tier']}")
    else:
        interactive_mode(model_slug=args.model_slug)


if __name__ == '__main__':
    main()
