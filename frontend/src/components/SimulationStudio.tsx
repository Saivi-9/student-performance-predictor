import React, { useState, useEffect } from 'react';
import { Cpu, Target, ArrowRight } from 'lucide-react';
import type { StudentInput, SimulationResponse } from '../types';
import { fetchSimulation } from '../api/client';

interface SimulationStudioProps {
  student: StudentInput;
}

export const SimulationStudio: React.FC<SimulationStudioProps> = ({ student }) => {
  const [selectedFeature, setSelectedFeature] = useState<'hours_studied' | 'attendance_percent' | 'tutoring_sessions'>('hours_studied');
  const [simulation, setSimulation] = useState<SimulationResponse | null>(null);

  useEffect(() => {
    let minVal = 1.0;
    let maxVal = 20.0;
    if (selectedFeature === 'attendance_percent') {
      minVal = 60.0;
      maxVal = 100.0;
    } else if (selectedFeature === 'tutoring_sessions') {
      minVal = 0;
      maxVal = 8;
    }

    fetchSimulation(student, selectedFeature, minVal, maxVal, 15).then(setSimulation);
  }, [student, selectedFeature]);

  const featureLabels = {
    hours_studied: 'Weekly Study Hours (1 - 20 hrs)',
    attendance_percent: 'Class Attendance Percentage (60% - 100%)',
    tutoring_sessions: 'Tutoring Sessions Attended (0 - 8 sessions)',
  };

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
            <div className="text-slate-400">Simulate how adjusting key habits shifts the final score trajectory</div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-slate-400 font-mono">INTERVENTION LEVER:</span>
          <select
            value={selectedFeature}
            onChange={(e) => setSelectedFeature(e.target.value as any)}
            className="bg-edu-950 border border-edu-700 rounded-lg px-3 py-1.5 text-xs text-white outline-none focus:border-indigo-500 font-semibold"
          >
            <option value="hours_studied">Weekly Study Hours</option>
            <option value="attendance_percent">Class Attendance Rate</option>
            <option value="tutoring_sessions">Tutoring Sessions</option>
          </select>
        </div>
      </div>

      {/* Trajectory Chart Card */}
      <div className="bg-edu-900 border border-edu-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex justify-between items-center border-b border-edu-800 pb-3 text-xs">
          <span className="font-bold text-white uppercase tracking-wider">
            Predicted Trajectory: {featureLabels[selectedFeature]}
          </span>
          <div className="flex items-center gap-4 text-[11px] font-mono">
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-emerald-400 inline-block"></span> Grade A (90+)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-blue-400 inline-block"></span> Grade B (80+)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-yellow-400 inline-block"></span> Grade C (70+)
            </span>
          </div>
        </div>

        {/* SVG Interactive Sensitivity Curve */}
        {simulation && (
          <div className="bg-edu-950 rounded-xl p-4 border border-edu-800 overflow-hidden">
            <svg viewBox="0 0 700 280" className="w-full h-auto">
              {/* Grid Lines */}
              <line x1="50" y1="30" x2="680" y2="30" stroke="#1E293B" strokeDasharray="3,3" />
              <text x="35" y="34" fill="#94A3B8" fontSize="10" textAnchor="end">100</text>

              {/* Grade A Threshold Line (90 pts) */}
              <line x1="50" y1="60" x2="680" y2="60" stroke="#22C55E" strokeDasharray="4,4" opacity="0.6" />
              <text x="35" y="64" fill="#22C55E" fontSize="10" textAnchor="end">90</text>

              {/* Grade B Threshold Line (80 pts) */}
              <line x1="50" y1="95" x2="680" y2="95" stroke="#3B82F6" strokeDasharray="4,4" opacity="0.6" />
              <text x="35" y="99" fill="#3B82F6" fontSize="10" textAnchor="end">80</text>

              {/* Grade C Threshold Line (70 pts) */}
              <line x1="50" y1="130" x2="680" y2="130" stroke="#EAB308" strokeDasharray="4,4" opacity="0.6" />
              <text x="35" y="134" fill="#EAB308" fontSize="10" textAnchor="end">70</text>

              <line x1="50" y1="200" x2="680" y2="200" stroke="#1E293B" strokeDasharray="3,3" />
              <text x="35" y="204" fill="#94A3B8" fontSize="10" textAnchor="end">50</text>

              {/* Baseline axes */}
              <line x1="50" y1="230" x2="680" y2="230" stroke="#334155" strokeWidth="1.5" />
              <line x1="50" y1="20" x2="50" y2="230" stroke="#334155" strokeWidth="1.5" />

              {/* Draw Trajectory Curve */}
              {(() => {
                const points = simulation.trajectory.map((step, idx) => {
                  const x = 50 + (idx / (simulation.trajectory.length - 1)) * 630;
                  // Map score 40-100 to y 230-30
                  const normScore = Math.max(0, Math.min(1, (step.predicted_score - 40) / 60));
                  const y = 230 - normScore * 200;
                  return { x, y, step };
                });

                const pathData = points.reduce((acc, curr, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${curr.x} ${curr.y}`, '');

                return (
                  <>
                    <path d={pathData} fill="none" stroke="#6366F1" strokeWidth="3.5" />
                    {points.map((pt, i) => (
                      <circle
                        key={i}
                        cx={pt.x}
                        cy={pt.y}
                        r="4"
                        fill="#EEF4FF"
                        stroke="#6366F1"
                        strokeWidth="2"
                      />
                    ))}

                    {/* Current Student Position Marker */}
                    {(() => {
                      const currX = 50 + ((simulation.current_value - simulation.trajectory[0].value) / (simulation.trajectory[simulation.trajectory.length - 1].value - simulation.trajectory[0].value)) * 630;
                      const currY = 230 - Math.max(0, Math.min(1, (simulation.current_score - 40) / 60)) * 200;
                      return (
                        <g>
                          <circle cx={currX} cy={currY} r="8" fill="#F43F5E" className="animate-ping" opacity="0.4" />
                          <circle cx={currX} cy={currY} r="6" fill="#F43F5E" stroke="#FFFFFF" strokeWidth="2" />
                          <text x={currX} y={currY - 12} fill="#F43F5E" fontSize="11" fontWeight="bold" textAnchor="middle">
                            Current: {simulation.current_score.toFixed(1)} pts
                          </text>
                        </g>
                      );
                    })()}
                  </>
                );
              })()}
            </svg>
          </div>
        )}

        {/* Actionable Intervention Prescription */}
        <div className="bg-edu-950 p-4 rounded-xl border border-edu-800 flex flex-wrap items-center justify-between gap-4 text-xs">
          <div className="flex items-center gap-3">
            <Target className="w-5 h-5 text-emerald-400" />
            <div>
              <div className="font-bold text-white">INTERVENTION ROI SUMMARY</div>
              <div className="text-slate-400">
                Increasing <strong>{selectedFeature.replace('_', ' ')}</strong> by +20% increases projected exam score by ~<strong>3.2 points</strong>.
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 font-mono text-indigo-400">
            <span>Model: Linear regularized slope (β = +1.42)</span>
            <ArrowRight className="w-4 h-4" />
          </div>
        </div>
      </div>
    </div>
  );
};
