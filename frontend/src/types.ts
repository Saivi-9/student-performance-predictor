export interface StudentInput {
  hours_studied: number;
  attendance_percent: number;
  previous_score: number;
  sleep_hours: number;
  tutoring_sessions: number;
  parental_involvement: 'Low' | 'Medium' | 'High';
  extracurricular: 'Yes' | 'No';
  internet_access: 'Yes' | 'No';
}

export interface PredictionResponse {
  predicted_score: number;
  grade_tier: 'Grade A' | 'Grade B' | 'Grade C' | 'Grade D' | 'Grade F';
  tier_label: string;
  status_icon: string;
  delta_vs_previous: number;
  effort_density_hrs_per_session: number;
  behavioral_diagnostics: {
    strengths: string[];
    recommendations: string[];
  };
  input_features: StudentInput;
}

export interface SimulationStep {
  value: number;
  predicted_score: number;
}

export interface SimulationResponse {
  feature_name: string;
  trajectory: SimulationStep[];
  current_value: number;
  current_score: number;
}
