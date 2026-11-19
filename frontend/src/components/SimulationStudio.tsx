import React, { useState, useEffect } from 'react';
import { Cpu, Target, ArrowRight } from 'lucide-react';
import type { StudentInput, SimulationResponse, ModelMetricEntry } from '../types';
import { fetchSimulation } from '../api/client';

interface SimulationStudioProps {
  student: StudentInput;
  models: ModelMetricEntry[];
  modelSlug: string | null;
  setModelSlug: (slug: string | null) => void;
}

const LEVER_CONFIG: Record<string, { label: string; min: number; max: number }> = {
  hours_studied: { label: 'Weekly Study Hours (1 - 30 hrs)', min: 1.0, max: 30.0 },
  attendance_percent: { label: 'Class Attendance Rate (50% - 100%)', min: 50.0, max: 100.0 },
  quiz_average: { label: 'Formative Quiz Average (40 - 100)', min: 40.0, max: 100.0 },
  assignment_completion_rate: { label: 'Assignment Completion (35% - 100%)', min: 35.0, max: 100.0 },
  sleep_hours: { label: 'Nightly Sleep Duration (4 - 10.5 hrs)', min: 4.0, max: 10.5 },
  stress_level: { label: 'Academic Stress Index (1 - 10)', min: 1.0, max: 10.0 },
  screen_time_hours: { label: 'Recreational Screen Time (0.5 - 10 hrs)', min: 0.5, max: 10.0 },
  practice_papers_solved: { label: 'Practice Papers Solved (0 - 15)', min: 0, max: 15 },
  tutoring_sessions: { label: 'Tutoring Sessions / Month (0 - 10)', min: 0, max: 10 },
};

export const SimulationStudio: React.FC<SimulationStudioProps> = ({
  student,
  models,
  modelSlug,
  setModelSlug,
}) => {
  const [selectedFeature, setSelectedFeature] = useState<string>('hours_studied');
  const [simulation, setSimulation] = useState<SimulationResponse | null>(null);

  useEffect(() => {
    const cfg = LEVER_CONFIG[selectedFeature] || LEVER_CONFIG.hours_studied;
    fetchSimulation(student, selectedFeature, cfg.min, cfg.max, 17, modelSlug).then(setSimulation);
  }, [student, selectedFeature, modelSlug]);

  const spanDelta =
    simulation && simulation.trajectory.length >= 2
      ? simulation.trajectory[simulation.trajectory.length - 1].predicted_score -
        simulation.trajectory[0].predicted_score
      : 0;

  return (
    <div className="space-y-6">
      {/* Banner */}
      <div className="bg-edu-900 border border-edu-800 rounded-2xl p-5 shadow-xl flex flex-wrap items-center justify-between gap-4 text-xs">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-edu-800 border border-edu-700 flex items-center justify-center text-indigo-400">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <div className="font-bold text-sm text-white">'WHAT-IF' INTERVENTION SENSITIVITY STUDIO</div>
            <div className="text-slate-400">
              Non-linear sensitivity curves computed live across {simulation?.model_used || 'active model'}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <label htmlFor="sim-lever-select" className="text-slate-400 font-mono">
              LEVER:
            </label>
            <select
              id="sim-lever-select"
              value={selectedFeature}
              onChange={(e) => setSelectedFeature(e.target.value)}
              className="bg-edu-950 border border-edu-700 rounded-lg px-3 py-1.5 text-xs text-white outline-none focus:border-indigo-500 font-semibold"
            >
              {Object.entries(LEVER_CONFIG).map(([key, cfg]) => (
                <option key={key} value={key}>
                  {cfg.label}
                </option>
              ))}
            </select>
          </div>

          <select
            aria-label="Simulation Model Engine"
            value={modelSlug || ''}
            onChange={(e) => setModelSlug(e.target.value || null)}
            className="bg-edu-950 border border-indigo-500/40 rounded-lg px-3 py-1.5 text-xs text-white outline-none focus:border-indigo-400 font-semibold"
          >
            <option value="">Auto Champion</option>
            {models.map((m) => (
              <option key={m.slug} value={m.slug}>
                {m.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Trajectory Chart Card */}
      <div className="bg-edu-900 border border-edu-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex flex-wrap justify-between items-center border-b border-edu-800 pb-3 text-xs gap-2">
          <span className="font-bold text-white uppercase tracking-wider">
            Predicted Trajectory: {LEVER_CONFIG[selectedFeature]?.label}
          </span>
          <div className="flex items-center gap-4 text-[11px] font-mono">
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-emerald-400 inline-block" /> Grade A (90+)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-blue-400 inline-block" /> Grade B (80+)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-yellow-400 inline-block" /> Grade C (70+)
            </span>
          </div>
        </div>

        {simulation && simulation.trajectory.length > 0 && (
          <div className="bg-edu-950 rounded-xl p-4 border border-edu-800 overflow-hidden">
            <svg viewBox="0 0 720 290" className="w-full h-auto" role="img" aria-label="What-If Intervention Trajectory Chart">
              <line x1="55" y1="30" x2="690" y2="30" stroke="#1E293B" strokeDasharray="3,3" />
              <text x="45" y="34" fill="#94A3B8" fontSize="10" textAnchor="end">100</text>

              <line x1="55" y1="63" x2="690" y2="63" stroke="#22C55E" strokeDasharray="4,4" opacity="0.6" />
              <text x="45" y="67" fill="#22C55E" fontSize="10" textAnchor="end">90</text>

              <line x1="55" y1="97" x2="690" y2="97" stroke="#3B82F6" strokeDasharray="4,4" opacity="0.6" />
              <text x="45" y="101" fill="#3B82F6" fontSize="10" textAnchor="end">80</text>

              <line x1="55" y1="130" x2="690" y2="130" stroke="#EAB308" strokeDasharray="4,4" opacity="0.6" />
              <text x="45" y="134" fill="#EAB308" fontSize="10" textAnchor="end">70</text>

              <line x1="55" y1="197" x2="690" y2="197" stroke="#1E293B" strokeDasharray="3,3" />
              <text x="45" y="201" fill="#94A3B8" fontSize="10" textAnchor="end">50</text>

              <line x1="55" y1="230" x2="690" y2="230" stroke="#334155" strokeWidth="1.5" />
              <line x1="55" y1="20" x2="55" y2="230" stroke="#334155" strokeWidth="1.5" />

              {(() => {
                const points = simulation.trajectory.map((step, idx) => {
                  const x = 55 + (idx / Math.max(1, simulation.trajectory.length - 1)) * 635;
                  const normScore = Math.max(0, Math.min(1, (step.predicted_score - 40) / 60));
                  const y = 230 - normScore * 200;
                  return { x, y, step };
                });

                const pathData = points.reduce(
                  (acc, curr, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${curr.x} ${curr.y}`,
                  ''
                );

                const firstVal = simulation.trajectory[0].value;
                const lastVal = simulation.trajectory[simulation.trajectory.length - 1].value;
                const clampedRatio = Math.max(
                  0,
                  Math.min(1, (simulation.current_value - firstVal) / Math.max(1e-5, lastVal - firstVal))
                );
                const currX = 55 + clampedRatio * 635;
                const currY = 230 - Math.max(0, Math.min(1, (simulation.current_score - 40) / 60)) * 200;

                return (
                  <>
                    <path d={pathData} fill="none" stroke="#6366F1" strokeWidth="3.5" />
                    {points.map((pt, i) => (
                      <g key={i}>
                        <circle
                          cx={pt.x}
                          cy={pt.y}
                          r="4"
                          fill="#EEF4FF"
                          stroke="#6366F1"
                          strokeWidth="2"
                        >
                          <title>{`${selectedFeature}: ${pt.step.value} -> ${pt.step.predicted_score} pts`}</title>
                        </circle>
                        {i % 4 === 0 && (
                          <text x={pt.x} y="252" fill="#94A3B8" fontSize="10" textAnchor="middle">
                            {pt.step.value}
                          </text>
                        )}
                      </g>
                    ))}

                    <g>
                      <circle cx={currX} cy={currY} r="9" fill="#F43F5E" className="animate-ping" opacity="0.35" />
                      <circle cx={currX} cy={currY} r="6" fill="#F43F5E" stroke="#FFFFFF" strokeWidth="2" />
                      <text x={currX} y={Math.max(18, currY - 12)} fill="#FDA4AF" fontSize="11" fontWeight="bold" textAnchor="middle">
                        Current ({simulation.current_value}): {simulation.current_score.toFixed(1)}
                      </text>
                    </g>
                  </>
                );
              })()}
            </svg>
          </div>
        )}

        {/* Dynamic ROI Summary */}
        <div className="bg-edu-950 p-4 rounded-xl border border-edu-800 flex flex-wrap items-center justify-between gap-4 text-xs">
          <div className="flex items-center gap-3">
            <Target className="w-5 h-5 text-emerald-400" />
            <div>
              <div className="font-bold text-white">EMPIRICAL INTERVENTION SWEEP SUMMARY</div>
              <div className="text-slate-400">
                Sweeping <strong>{selectedFeature.replace(/_/g, ' ')}</strong> across its full range shifts the forecast by{' '}
                <strong className={spanDelta >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                  {spanDelta >= 0 ? `+${spanDelta.toFixed(1)}` : spanDelta.toFixed(1)} pts
                </strong>
                .
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 font-mono text-indigo-400">
            <span>Engine: {simulation?.model_used || 'Champion Pipeline'}</span>
            <ArrowRight className="w-4 h-4" />
          </div>
        </div>
      </div>
    </div>
  );
};
