import React, { useState, useEffect } from 'react';
import { 
  BookOpen, Calendar, Clock, Award, 
  Users, Activity, ArrowUpRight, ArrowDownRight, Lightbulb 
} from 'lucide-react';
import type { StudentInput, PredictionResponse } from '../types';
import { fetchPrediction } from '../api/client';

interface StudentForecasterProps {
  student: StudentInput;
  setStudent: React.Dispatch<React.SetStateAction<StudentInput>>;
}

export const StudentForecaster: React.FC<StudentForecasterProps> = ({ student, setStudent }) => {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    setLoading(true);
    fetchPrediction(student)
      .then(setPrediction)
      .finally(() => setLoading(false));
  }, [student]);

  const getTierColor = (tier: string) => {
    switch (tier) {
      case 'Grade A': return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
      case 'Grade B': return 'text-blue-400 bg-blue-500/10 border-blue-500/30';
      case 'Grade C': return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30';
      case 'Grade D': return 'text-orange-400 bg-orange-500/10 border-orange-500/30';
      default: return 'text-rose-400 bg-rose-500/10 border-rose-500/30';
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-edu-900 border border-edu-800 rounded-2xl p-5 shadow-xl flex flex-wrap items-center justify-between gap-4 text-xs">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-edu-800 border border-edu-700 flex items-center justify-center text-indigo-400">
            <BookOpen className="w-5 h-5" />
          </div>
          <div>
            <div className="font-bold text-sm text-white">STUDENT ACADEMIC PROFILE & LIVE SIMULATOR</div>
            <div className="text-slate-400 font-mono">Real-time inference using leak-free Ridge Regularized Regression</div>
          </div>
        </div>

        <div className="flex items-center gap-4 text-slate-300 font-mono">
          <div><span className="text-slate-500">TRAINING COHORT:</span> 1,000 RECORDS</div>
          <div><span className="text-slate-500">CV SCHEME:</span> 5-FOLD CV</div>
        </div>
      </div>

      {/* Main Grid: Inputs on Left (6 Cols), Output Card on Right (6 Cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Sliders (6 Cols) */}
        <div className="lg:col-span-6 space-y-4">
          <div className="bg-edu-900 border border-edu-800 rounded-2xl p-5 shadow-xl space-y-5">
            <h3 className="text-sm font-extrabold uppercase tracking-wider text-slate-300 flex items-center gap-2 border-b border-edu-800 pb-2">
              <Activity className="w-4 h-4 text-indigo-400" />
              Behavioral & Academic Parameters
            </h3>

            {/* Hours Studied */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                  <BookOpen className="w-3.5 h-3.5 text-indigo-400" /> Weekly Study Hours
                </span>
                <span className="font-mono font-bold text-indigo-300">{student.hours_studied.toFixed(1)} hrs/week</span>
              </div>
              <input
                type="range"
                min="1.0"
                max="25.0"
                step="0.5"
                value={student.hours_studied}
                onChange={(e) => setStudent({ ...student, hours_studied: parseFloat(e.target.value) })}
                className="w-full accent-indigo-500 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500">
                <span>1.0 hr (Minimal)</span>
                <span>10.0 hrs (Target)</span>
                <span>25.0 hrs (Intensive)</span>
              </div>
            </div>

            {/* Attendance Rate */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                  <Calendar className="w-3.5 h-3.5 text-emerald-400" /> Lecture Attendance Rate
                </span>
                <span className="font-mono font-bold text-emerald-300">{student.attendance_percent.toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="40.0"
                max="100.0"
                step="1.0"
                value={student.attendance_percent}
                onChange={(e) => setStudent({ ...student, attendance_percent: parseFloat(e.target.value) })}
                className="w-full accent-emerald-500 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500">
                <span>40% (High Absence Risk)</span>
                <span>80% (Threshold)</span>
                <span>100% (Perfect)</span>
              </div>
            </div>

            {/* Prior Midterm Score */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                  <Award className="w-3.5 h-3.5 text-blue-400" /> Prior Midterm Exam Baseline
                </span>
                <span className="font-mono font-bold text-blue-300">{student.previous_score.toFixed(0)} pts</span>
              </div>
              <input
                type="range"
                min="30.0"
                max="100.0"
                step="1.0"
                value={student.previous_score}
                onChange={(e) => setStudent({ ...student, previous_score: parseFloat(e.target.value) })}
                className="w-full accent-blue-500 cursor-pointer"
              />
            </div>

            {/* Sleep Hours & Tutoring Sessions */}
            <div className="grid grid-cols-2 gap-4 pt-1">
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-semibold flex items-center gap-1">
                    <Clock className="w-3.5 h-3.5 text-purple-400" /> Sleep (hrs)
                  </span>
                  <span className="font-mono font-bold text-purple-300">{student.sleep_hours.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="4.0"
                  max="11.0"
                  step="0.5"
                  value={student.sleep_hours}
                  onChange={(e) => setStudent({ ...student, sleep_hours: parseFloat(e.target.value) })}
                  className="w-full accent-purple-500 cursor-pointer"
                />
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-semibold flex items-center gap-1">
                    <Users className="w-3.5 h-3.5 text-yellow-400" /> Tutoring
                  </span>
                  <span className="font-mono font-bold text-yellow-300">{student.tutoring_sessions} sessions</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="1"
                  value={student.tutoring_sessions}
                  onChange={(e) => setStudent({ ...student, tutoring_sessions: parseInt(e.target.value) })}
                  className="w-full accent-yellow-500 cursor-pointer"
                />
              </div>
            </div>

            {/* Categoricals (Parental, Extracurricular, Internet) */}
            <div className="grid grid-cols-3 gap-3 pt-2 text-xs">
              <div>
                <label className="text-slate-400 block mb-1">Parental Support</label>
                <select
                  value={student.parental_involvement}
                  onChange={(e) => setStudent({ ...student, parental_involvement: e.target.value as any })}
                  className="w-full bg-edu-950 border border-edu-800 rounded-lg p-2 text-white outline-none focus:border-indigo-500"
                >
                  <option value="Low">Low</option>
                  <option value="Medium">Medium</option>
                  <option value="High">High</option>
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Extracurriculars</label>
                <select
                  value={student.extracurricular}
                  onChange={(e) => setStudent({ ...student, extracurricular: e.target.value as any })}
                  className="w-full bg-edu-950 border border-edu-800 rounded-lg p-2 text-white outline-none focus:border-indigo-500"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Home Broadband</label>
                <select
                  value={student.internet_access}
                  onChange={(e) => setStudent({ ...student, internet_access: e.target.value as any })}
                  className="w-full bg-edu-950 border border-edu-800 rounded-lg p-2 text-white outline-none focus:border-indigo-500"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Outcome & Badges (6 Cols) */}
        <div className="lg:col-span-6 space-y-4">
          <div className="bg-edu-900 border border-edu-800 rounded-2xl p-6 shadow-xl space-y-5">
            <div className="flex items-center justify-between border-b border-edu-800 pb-3">
              <span className="text-xs uppercase font-extrabold tracking-wider text-slate-400">
                AI PREDICTIVE OUTCOME
              </span>
              <span className="text-xs font-mono text-indigo-400">
                {loading ? 'CALCULATING...' : 'RIDGE REGRESSION MODEL'}
              </span>
            </div>

            {prediction && (
              <>
                {/* Large Metric Display with Tier Badge */}
                <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-edu-950 border border-edu-800 rounded-2xl p-5">
                  <div>
                    <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold block">
                      Projected Final Exam Score
                    </span>
                    <div className="text-4xl sm:text-5xl font-black text-white mt-1">
                      {prediction.predicted_score.toFixed(1)} <span className="text-lg text-slate-500 font-normal">/ 100</span>
                    </div>
                  </div>

                  <div className="text-right flex flex-col items-end">
                    <span className={`px-4 py-1.5 rounded-full text-xs font-bold border flex items-center gap-1.5 ${getTierColor(prediction.grade_tier)}`}>
                      <span>{prediction.status_icon}</span>
                      <span>{prediction.grade_tier}</span>
                    </span>
                    <span className="text-xs text-slate-400 mt-1 max-w-[200px] text-right font-medium">
                      {prediction.tier_label}
                    </span>
                  </div>
                </div>

                {/* Secondary Indicators */}
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="bg-edu-950 border border-edu-800 rounded-xl p-3.5">
                    <span className="text-slate-400 text-[10px] uppercase font-bold tracking-wider block">Trajectory Delta</span>
                    <div className="flex items-center gap-1 text-lg font-bold mt-0.5">
                      {prediction.delta_vs_previous >= 0 ? (
                        <>
                          <ArrowUpRight className="w-5 h-5 text-emerald-400" />
                          <span className="text-emerald-400">+{prediction.delta_vs_previous.toFixed(1)} pts</span>
                        </>
                      ) : (
                        <>
                          <ArrowDownRight className="w-5 h-5 text-rose-400" />
                          <span className="text-rose-400">{prediction.delta_vs_previous.toFixed(1)} pts</span>
                        </>
                      )}
                    </div>
                    <span className="text-[11px] text-slate-500 font-mono">vs Prior Midterm Baseline</span>
                  </div>

                  <div className="bg-edu-950 border border-edu-800 rounded-xl p-3.5">
                    <span className="text-slate-400 text-[10px] uppercase font-bold tracking-wider block">Effort Density</span>
                    <div className="text-lg font-bold text-indigo-300 mt-0.5">
                      {prediction.effort_density_hrs_per_session.toFixed(1)} <span className="text-xs text-slate-500">hrs/session</span>
                    </div>
                    <span className="text-[11px] text-slate-500 font-mono">Study Hours per Tutoring Unit</span>
                  </div>
                </div>

                {/* Diagnostics & Advisor Recommendations */}
                <div className="bg-edu-950/90 border border-edu-800 rounded-xl p-4 space-y-2.5 text-xs">
                  <div className="flex items-center gap-1.5 text-indigo-400 font-bold">
                    <Lightbulb className="w-4 h-4" /> Academic Advisory Diagnostics
                  </div>

                  <div className="space-y-1.5 pt-1">
                    {prediction.behavioral_diagnostics.recommendations.map((rec, i) => (
                      <div key={i} className="flex items-start gap-2 text-slate-300">
                        <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 mt-1.5 flex-shrink-0"></span>
                        <span>{rec}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
