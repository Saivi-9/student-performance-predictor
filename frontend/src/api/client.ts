import type {
  StudentInput,
  PredictionResponse,
  SimulationResponse,
  BenchmarkReport,
  HealthStatus,
  PretrainedCatalogEntry,
  HubSearchItem,
  PretrainedBenchmarkEntry,
  BatchPredictionResponse,
  DatasetSummaryResponse,
} from '../types';

const API_BASE_URL =
  typeof window !== 'undefined' && window.location.port === '8001'
    ? ''
    : 'http://localhost:8001';

function computeOfflineScore(s: StudentInput): number {
  const sleepPenalty = 0.7 * Math.abs(s.sleep_hours - 8.0);
  const ordMap = { Low: -1.8, Medium: 0.4, High: 2.2 };
  const raw =
    0.34 * s.previous_score +
    0.24 * s.quiz_average +
    4.6 * Math.sqrt(Math.max(s.hours_studied, 0)) +
    0.18 * s.attendance_percent +
    0.15 * s.assignment_completion_rate +
    0.7 * s.practice_papers_solved +
    0.45 * s.tutoring_sessions -
    sleepPenalty -
    0.65 * s.stress_level -
    0.65 * s.screen_time_hours +
    ordMap[s.parental_involvement] +
    ordMap[s.motivation_level] -
    8.5;
  return Math.max(0, Math.min(100, Math.round(raw * 10) / 10));
}

export async function fetchHealth(): Promise<HealthStatus | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function fetchPrediction(
  student: StudentInput,
  modelSlug?: string | null
): Promise<PredictionResponse> {
  try {
    const qs = modelSlug ? `?model_slug=${encodeURIComponent(modelSlug)}` : '';
    const res = await fetch(`${API_BASE_URL}/v1/predict${qs}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(student),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('Backend offline, using client-side v2 approximation:', err);
    const score = computeOfflineScore(student);

    let tier: PredictionResponse['grade_tier'] = 'Grade F';
    let label = 'Academic Risk / Intensive Intervention Required';
    let icon = '🚨';
    if (score >= 90) {
      tier = 'Grade A';
      label = "Dean's Honors / Outstanding Mastery";
      icon = '🌟';
    } else if (score >= 80) {
      tier = 'Grade B';
      label = 'Very Good / Strong Conceptual Grasp';
      icon = '📘';
    } else if (score >= 70) {
      tier = 'Grade C';
      label = 'Good / Satisfactory Standing';
      icon = '📗';
    } else if (score >= 60) {
      tier = 'Grade D';
      label = 'Passing / Needs Reinforcement';
      icon = '⚠️';
    }

    return {
      predicted_score: score,
      confidence_interval: [Math.max(0, Math.round((score - 2.0) * 10) / 10), Math.min(100, Math.round((score + 2.0) * 10) / 10)],
      model_used: modelSlug || 'Offline Local Estimator',
      grade_tier: tier,
      tier_label: label,
      status_icon: icon,
      delta_vs_previous: Math.round((score - student.previous_score) * 10) / 10,
      effort_density_hrs_per_session: Math.round((student.hours_studied / (student.tutoring_sessions + 1)) * 100) / 100,
      engineered_features: {
        study_efficiency: Math.round((student.hours_studied * (student.attendance_percent / 100)) * 100) / 100,
        academic_momentum: Math.round((0.55 * student.quiz_average + 0.45 * student.previous_score) * 10) / 10,
        sleep_deficit: Math.round(Math.abs(student.sleep_hours - 8.0) * 100) / 100,
        engagement_score: Math.round((0.5 * (student.attendance_percent / 100) + 0.5 * (student.assignment_completion_rate / 100)) * 1000) / 1000,
        study_sleep_balance: Math.round((student.hours_studied / Math.max(student.sleep_hours, 1)) * 100) / 100,
        digital_distraction_ratio: Math.round((student.screen_time_hours / (student.hours_studied / 7 + 0.5)) * 100) / 100,
        formative_delta: Math.round((student.quiz_average - student.previous_score) * 10) / 10,
        wellbeing_index: Math.round(((10 - student.stress_level) / 10) * 100) / 100,
      },
      feature_contributions: [],
      behavioral_diagnostics: {
        strengths: [`Prior score anchor: ${student.previous_score.toFixed(0)} pts`],
        recommendations: ['Start backend API server for full LightGBM / Ensemble live predictions & attributions.'],
      },
      input_features: student,
    };
  }
}

export async function fetchSimulation(
  student: StudentInput,
  featureName: string,
  minVal: number,
  maxVal: number,
  steps: number = 15,
  modelSlug?: string | null
): Promise<SimulationResponse> {
  try {
    const slugParam = modelSlug ? `&model_slug=${encodeURIComponent(modelSlug)}` : '';
    const url = `${API_BASE_URL}/v1/simulate?feature_name=${encodeURIComponent(featureName)}&min_val=${minVal}&max_val=${maxVal}&steps=${steps}${slugParam}`;
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(student),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('Backend offline, using client-side simulation:', err);
    const stepSize = (maxVal - minVal) / (steps - 1);
    const trajectory = [];
    for (let i = 0; i < steps; i++) {
      const val = minVal + i * stepSize;
      const mutated = { ...student, [featureName]: val };
      trajectory.push({ value: Math.round(val * 100) / 100, predicted_score: computeOfflineScore(mutated) });
    }
    return {
      feature_name: featureName,
      model_used: modelSlug || 'Offline Estimator',
      trajectory,
      current_value: Number((student as any)[featureName] ?? minVal),
      current_score: computeOfflineScore(student),
    };
  }
}

export async function fetchBenchmarks(): Promise<BenchmarkReport | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/v1/benchmarks`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function fetchDatasetSummary(sampleLimit: number = 12): Promise<DatasetSummaryResponse | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/v1/dataset/summary?sample_limit=${sampleLimit}`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function fetchPretrainedCatalog(): Promise<PretrainedCatalogEntry[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/v1/pretrained/catalog`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.catalog || [];
  } catch {
    return [];
  }
}

export async function searchPretrainedHub(
  query: string,
  kind: 'models' | 'datasets' = 'models',
  limit: number = 10
): Promise<HubSearchItem[]> {
  const res = await fetch(
    `${API_BASE_URL}/v1/pretrained/search?query=${encodeURIComponent(query)}&kind=${kind}&limit=${limit}`
  );
  if (!res.ok) throw new Error(`Hub search failed: ${res.status}`);
  const data = await res.json();
  return data.results || [];
}

export async function runPretrainedBenchmark(refresh: boolean = false): Promise<PretrainedBenchmarkEntry[]> {
  const res = await fetch(`${API_BASE_URL}/v1/pretrained/benchmark?refresh=${refresh}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Pretrained benchmark failed: ${res.status}`);
  const data = await res.json();
  return data.benchmarks || [];
}

export async function runPretrainedPrediction(student: StudentInput): Promise<{
  repo_id: string;
  predicted_score: number;
  grade_tier: string;
  tier_label: string;
  status_icon: string;
}> {
  const res = await fetch(`${API_BASE_URL}/v1/pretrained/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(student),
  });
  if (!res.ok) throw new Error(`Pretrained prediction failed: ${res.status}`);
  return await res.json();
}

export async function fetchBatchPrediction(
  students: StudentInput[],
  modelSlug?: string | null
): Promise<BatchPredictionResponse> {
  const res = await fetch(`${API_BASE_URL}/v1/predict/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ students, model_slug: modelSlug || null }),
  });
  if (!res.ok) throw new Error(`Batch prediction failed: ${res.status}`);
  return await res.json();
}
