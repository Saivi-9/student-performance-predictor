import type { StudentInput, PredictionResponse, SimulationResponse } from '../types';

const API_BASE_URL = 'http://localhost:8001';

export async function fetchPrediction(student: StudentInput): Promise<PredictionResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/v1/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(student),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('Backend offline, using client-side calculation:', err);
    // Ridge champion model formula approximation
    const raw = 0.48 * student.previous_score + 
                1.42 * student.hours_studied + 
                0.22 * student.attendance_percent + 
                0.45 * student.tutoring_sessions + 
                (student.parental_involvement === 'High' ? 2.5 : (student.parental_involvement === 'Low' ? -2.0 : 0.5));
    const score = Math.max(0, Math.min(100, Math.round(raw * 10) / 10));

    let tier: PredictionResponse['grade_tier'] = 'Grade F';
    let label = 'Academic Risk / Intervention Recommended';
    let icon = '🚨';
    if (score >= 90) { tier = 'Grade A'; label = "Dean's Honors / Outstanding Mastery"; icon = '🌟'; }
    else if (score >= 80) { tier = 'Grade B'; label = 'Very Good / Strong Conceptual Grasp'; icon = '📘'; }
    else if (score >= 70) { tier = 'Grade C'; label = 'Good / Satisfactory Standing'; icon = '📗'; }
    else if (score >= 60) { tier = 'Grade D'; label = 'Passing / Needs Reinforcement'; icon = '⚠️'; }

    return {
      predicted_score: score,
      grade_tier: tier,
      tier_label: label,
      status_icon: icon,
      delta_vs_previous: Math.round((score - student.previous_score) * 10) / 10,
      effort_density_hrs_per_session: Math.round((student.hours_studied / (student.tutoring_sessions + 1)) * 10) / 10,
      behavioral_diagnostics: {
        strengths: [
          `Prior exam foundation of ${student.previous_score.toFixed(0)} points`,
          `Attendance presence at ${student.attendance_percent.toFixed(0)}%`,
        ],
        recommendations: student.hours_studied < 8.0 
          ? ['Increasing weekly self-study by +2 hours unlocks potential for higher grades.'] 
          : ['Current study pacing is well-calibrated.']
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
  steps: number = 15
): Promise<SimulationResponse> {
  try {
    const url = `${API_BASE_URL}/v1/simulate?feature_name=${featureName}&min_val=${minVal}&max_val=${maxVal}&steps=${steps}`;
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
    const baseCopy = { ...student };

    for (let i = 0; i < steps; i++) {
      const val = minVal + i * stepSize;
      const mutated = { ...baseCopy, [featureName]: val };
      const raw = 0.48 * mutated.previous_score + 
                  1.42 * mutated.hours_studied + 
                  0.22 * mutated.attendance_percent + 
                  0.45 * mutated.tutoring_sessions + 
                  (mutated.parental_involvement === 'High' ? 2.5 : (mutated.parental_involvement === 'Low' ? -2.0 : 0.5));
      const s = Math.max(0, Math.min(100, Math.round(raw * 10) / 10));
      trajectory.push({ value: Math.round(val * 10) / 10, predicted_score: s });
    }

    const currentScore = trajectory.find(t => Math.abs(t.value - (student as any)[featureName]) < stepSize)?.predicted_score || 80;

    return {
      feature_name: featureName,
      trajectory,
      current_value: (student as any)[featureName],
      current_score: currentScore,
    };
  }
}
