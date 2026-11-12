export type OrdinalLevel = 'Low' | 'Medium' | 'High';
export type BinaryFlag = 'Yes' | 'No';
export type ParentEducation = 'High School' | 'Bachelor' | 'Master+';
export type SchoolType = 'Public' | 'Private';
export type LearningStyle = 'Visual' | 'Auditory' | 'Kinesthetic' | 'Reading/Writing';

export interface StudentInput {
  hours_studied: number;
  attendance_percent: number;
  previous_score: number;
  quiz_average: number;
  assignment_completion_rate: number;
  practice_papers_solved: number;
  sleep_hours: number;
  tutoring_sessions: number;
  physical_activity_hours: number;
  screen_time_hours: number;
  stress_level: number;
  commute_minutes: number;
  parental_involvement: OrdinalLevel;
  motivation_level: OrdinalLevel;
  access_to_resources: OrdinalLevel;
  parental_education: ParentEducation;
  extracurricular: BinaryFlag;
  internet_access: BinaryFlag;
  school_type: SchoolType;
  learning_style: LearningStyle;
}

export interface FactorContribution {
  feature: string;
  label: string;
  impact: number;
  direction: 'positive' | 'negative' | string;
}

export interface PredictionResponse {
  predicted_score: number;
  confidence_interval: number[];
  model_used: string;
  grade_tier: 'Grade A' | 'Grade B' | 'Grade C' | 'Grade D' | 'Grade F';
  tier_label: string;
  status_icon: string;
  delta_vs_previous: number;
  effort_density_hrs_per_session: number;
  engineered_features: Record<string, number>;
  feature_contributions: FactorContribution[];
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
  model_used: string;
  trajectory: SimulationStep[];
  current_value: number;
  current_score: number;
}

export interface ModelMetricEntry {
  name: string;
  slug: string;
  cv_mae: number;
  cv_rmse: number;
  cv_r2: number;
  val_mae: number;
  val_rmse: number;
  val_r2: number;
  test_mae: number;
  test_rmse: number;
  test_r2: number;
  train_time_sec: number;
  is_champion: boolean;
}

export interface FeatureImportanceEntry {
  feature: string;
  importance_mean: number;
  importance_std: number;
}

export interface DiagnosticSample {
  actual: number;
  predicted: number;
  residual: number;
}

export interface BenchmarkReport {
  champion_model: string;
  champion_slug: string;
  dataset_summary: {
    total_rows: number;
    raw_features: number;
    engineered_features: number;
    total_pipeline_features: number;
    train_rows: number;
    val_rows: number;
    test_rows: number;
    missing_values_handled?: Record<string, number>;
  };
  test_metrics: {
    MAE: number;
    RMSE: number;
    R2: number;
  };
  models: ModelMetricEntry[];
  feature_importances: FeatureImportanceEntry[];
  diagnostic_samples: DiagnosticSample[];
}

export interface HealthStatus {
  status: string;
  service: string;
  version: string;
  model_loaded: boolean;
  champion_model: string;
  dataset_rows: number;
  total_features: number;
  models_available: number;
}

export interface PretrainedCatalogEntry {
  repo_id: string;
  filename: string;
  framework: string;
  task: string;
  dataset: string;
  supports_live_inference: boolean;
  url: string;
}

export interface HubSearchItem {
  id: string;
  downloads: number;
  likes: number;
  pipeline_tag?: string;
  tags: string[];
  url: string;
}

export interface PretrainedBenchmarkEntry {
  repo_id: string;
  model_type?: string;
  benchmark_dataset?: string;
  target_scale?: string;
  mae?: number;
  rmse?: number;
  r2?: number;
  features_used?: number;
  url?: string;
  error?: string;
}

export interface BatchPredictionResponse {
  count: number;
  model_used: string;
  average_score: number;
  at_risk_count: number;
  predictions: {
    index: number;
    predicted_score: number;
    grade_tier: string;
    academic_momentum: number;
    engagement_score: number;
  }[];
}

export interface DatasetSummaryResponse {
  dataset_summary: BenchmarkReport['dataset_summary'];
  sample_records: Record<string, any>[];
  external_sources: {
    name: string;
    source: string;
    rows: number;
    columns: number;
  }[];
}
