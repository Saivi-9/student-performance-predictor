import React, { useState, useEffect } from 'react';
import {
  BookOpen,
  Calendar,
  Clock,
  Award,
  Users,
  Activity,
  ArrowUpRight,
  ArrowDownRight,
  Lightbulb,
  Sliders,
  BrainCircuit,
  Sparkles,
} from 'lucide-react';
import type { StudentInput, PredictionResponse, ModelMetricEntry } from '../types';
import { fetchPrediction } from '../api/client';

interface StudentForecasterProps {
  student: StudentInput;
  setStudent: React.Dispatch<React.SetStateAction<StudentInput>>;
  models: ModelMetricEntry[];
  modelSlug: string | null;
  setModelSlug: (slug: string | null) => void;
}

const PRESET_PROFILES: { label: string; tag: string; data: Partial<StudentInput> }[] = [
  {
    label: 'High Achiever',
    tag: 'Grade A',
    data: {
      previous_score: 91,
      quiz_average: 93,
      hours_studied: 18,
      attendance_percent: 97,
      assignment_completion_rate: 96,
      practice_papers_solved: 9,
      sleep_hours: 8.0,
      tutoring_sessions: 3,
      stress_level: 3,
      screen_time_hours: 1.5,
      parental_involvement: 'High',
      motivation_level: 'High',
      access_to_resources: 'High',
    },
  },
  {
    label: 'Balanced Learner',
    tag: 'Grade B',
    data: {
      previous_score: 74,
      quiz_average: 78,
      hours_studied: 12,
      attendance_percent: 88,
      assignment_completion_rate: 85,
      practice_papers_solved: 5,
      sleep_hours: 7.5,
      tutoring_sessions: 2,
      stress_level: 4,
      screen_time_hours: 3.0,
      parental_involvement: 'Medium',
      motivation_level: 'Medium',
      access_to_resources: 'Medium',
    },
  },
  {
    label: 'Burnout / Overloaded',
    tag: 'Sleep/Stress',
    data: {
      previous_score: 76,
      quiz_average: 71,
      hours_studied: 21,
      attendance_percent: 84,
      assignment_completion_rate: 79,
      practice_papers_solved: 6,
      sleep_hours: 4.5,
      tutoring_sessions: 4,
      stress_level: 9,
      screen_time_hours: 5.5,
      commute_minutes: 65,
      motivation_level: 'Medium',
    },
  },
  {
    label: 'At-Risk Student',
    tag: 'Intervention',
    data: {
      previous_score: 52,
      quiz_average: 54,
      hours_studied: 4.5,
      attendance_percent: 62,
      assignment_completion_rate: 58,
      practice_papers_solved: 1,
      sleep_hours: 5.5,
      tutoring_sessions: 0,
      stress_level: 8,
      screen_time_hours: 7.0,
      parental_involvement: 'Low',
      motivation_level: 'Low',
      access_to_resources: 'Low',
      internet_access: 'No',
    },
  },
];

export const StudentForecaster: React.FC<StudentForecasterProps> = ({
  student,
  setStudent,
  models,
  modelSlug,
  setModelSlug,
}) => {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const requestKey = JSON.stringify({ student, modelSlug });
  const [resolvedKey, setResolvedKey] = useState<string>('');
  const loading = resolvedKey !== requestKey;

  useEffect(() => {
    let active = true;
    fetchPrediction(student, modelSlug)
      .then((res) => {
        if (active) {
          setPrediction(res);
          setResolvedKey(requestKey);
        }
      })
      .catch(() => {
        if (active) setResolvedKey(requestKey);
      });
    return () => {
      active = false;
    };
  }, [student, modelSlug, requestKey]);

  const getTierColor = (tier: string) => {
    switch (tier) {
      case 'Grade A':
        return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
      case 'Grade B':
        return 'text-blue-400 bg-blue-500/10 border-blue-500/30';
      case 'Grade C':
        return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30';
      case 'Grade D':
        return 'text-orange-400 bg-orange-500/10 border-orange-500/30';
      default:
        return 'text-rose-400 bg-rose-500/10 border-rose-500/30';
    }
  };

  const applyPreset = (patch: Partial<StudentInput>) => {
    setStudent((prev) => ({ ...prev, ...patch }));
  };

  return (
    <div className="space-y-6">
      {/* Top Control & Model Switcher Bar */}
      <div className="bg-edu-900 border border-edu-800 rounded-2xl p-5 shadow-xl flex flex-wrap items-center justify-between gap-4 text-xs">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-edu-800 border border-edu-700 flex items-center justify-center text-indigo-400">
            <BrainCircuit className="w-5 h-5" />
          </div>
          <div>
            <div className="font-bold text-sm text-white">
              REAL-TIME MULTI-MODEL FORECASTER (20 RAW + 8 ENGINEERED FEATURES)
            </div>
            <div className="text-slate-400 font-mono">
              Every slider & selector recalculates engineered features and factor attributions live
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <label htmlFor="active-model-select" className="text-slate-400 font-mono">
              ENGINE:
            </label>
            <select
              id="active-model-select"
              value={modelSlug || ''}
              onChange={(e) => setModelSlug(e.target.value || null)}
              className="bg-edu-950 border border-indigo-500/40 rounded-lg px-3 py-1.5 text-xs text-white outline-none focus:border-indigo-400 font-semibold"
            >
              <option value="">Auto Champion Model</option>
              {models.map((m) => (
                <option key={m.slug} value={m.slug}>
                  {m.name} (R² {m.test_r2.toFixed(4)})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Quick Archetype Presets */}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-slate-400 font-mono mr-1">PRESET PROFILES:</span>
        {PRESET_PROFILES.map((preset) => (
          <button
            key={preset.label}
            type="button"
            onClick={() => applyPreset(preset.data)}
            className="px-3 py-1.5 rounded-lg bg-edu-900 hover:bg-edu-800 border border-edu-800 hover:border-indigo-500/50 text-slate-200 transition flex items-center gap-2"
          >
            <span className="font-semibold">{preset.label}</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/15 text-indigo-300 font-mono">
              {preset.tag}
            </span>
          </button>
        ))}
      </div>

      {/* Main Split Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: All 20 Input Controls (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          {/* Panel 1: Academic Assessment */}
          <div className="bg-edu-900 border border-edu-800 rounded-2xl p-5 shadow-xl space-y-4">
            <h3 className="text-xs font-extrabold uppercase tracking-wider text-slate-300 flex items-center gap-2 border-b border-edu-800 pb-2">
              <Activity className="w-4 h-4 text-indigo-400" />
              1. Academic Performance & Coursework (6 Features)
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <label htmlFor="prior-score-range" className="text-slate-300 font-semibold">
                    Prior Exam Score
                  </label>
                  <span className="font-mono font-bold text-blue-300">{student.previous_score.toFixed(0)} pts</span>
                </div>
                <input
                  id="prior-score-range"
                  type="range"
                  min="30"
                  max="100"
                  step="1"
                  value={student.previous_score}
                  onChange={(e) => setStudent({ ...student, previous_score: parseFloat(e.target.value) })}
                  className="w-full cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <label htmlFor="quiz-avg-range" className="text-slate-300 font-semibold">
                    Formative Quiz Average
                  </label>
                  <span className="font-mono font-bold text-cyan-300">{student.quiz_average.toFixed(0)} pts</span>
                </div>
                <input
                  id="quiz-avg-range"
                  type="range"
                  min="30"
                  max="100"
                  step="1"
                  value={student.quiz_average}
                  onChange={(e) => setStudent({ ...student, quiz_average: parseFloat(e.target.value) })}
                  className="w-full cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <label htmlFor="study-hrs-range" className="text-slate-300 font-semibold flex items-center gap-1">
                    <BookOpen className="w-3.5 h-3.5 text-indigo-400" /> Weekly Study Hours
                  </label>
                  <span className="font-mono font-bold text-indigo-300">{student.hours_studied.toFixed(1)} hrs</span>
                </div>
                <input
                  id="study-hrs-range"
                  type="range"
                  min="1"
                  max="35"
                  step="0.5"
                  value={student.hours_studied}
                  onChange={(e) => setStudent({ ...student, hours_studied: parseFloat(e.target.value) })}
                  className="w-full cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <label htmlFor="attendance-range" className="text-slate-300 font-semibold flex items-center gap-1">
                    <Calendar className="w-3.5 h-3.5 text-emerald-400" /> Attendance Rate
                  </label>
                  <span className="font-mono font-bold text-emerald-300">{student.attendance_percent.toFixed(0)}%</span>
                </div>
                <input
                  id="attendance-range"
                  type="range"
                  min="40"
                  max="100"
                  step="1"
                  value={student.attendance_percent}
                  onChange={(e) => setStudent({ ...student, attendance_percent: parseFloat(e.target.value) })}
                  className="w-full cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <label htmlFor="assign-range" className="text-slate-300 font-semibold">
                    Assignment Completion
                  </label>
                  <span className="font-mono font-bold text-teal-300">{student.assignment_completion_rate.toFixed(0)}%</span>
                </div>
                <input
                  id="assign-range"
                  type="range"
                  min="25"
                  max="100"
                  step="1"
                  value={student.assignment_completion_rate}
                  onChange={(e) => setStudent({ ...student, assignment_completion_rate: parseFloat(e.target.value) })}
                  className="w-full cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <label htmlFor="papers-range" className="text-slate-300 font-semibold">
                    Practice Papers Solved
                  </label>
                  <span className="font-mono font-bold text-amber-300">{student.practice_papers_solved} papers</span>
                </div>
                <input
                  id="papers-range"
                  type="range"
                  min="0"
                  max="15"
                  step="1"
                  value={student.practice_papers_solved}
                  onChange={(e) => setStudent({ ...student, practice_papers_solved: parseInt(e.target.value, 10) })}
                  className="w-full cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Panel 2: Wellbeing & Habits */}
          <div className="bg-edu-900 border border-edu-800 rounded-2xl p-5 shadow-xl space-y-4">
            <h3 className="text-xs font-extrabold uppercase tracking-wider text-slate-300 flex items-center gap-2 border-b border-edu-800 pb-2">
              <Clock className="w-4 h-4 text-purple-400" />
              2. Wellbeing, Cognitive Load & Habits (6 Features)
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <label htmlFor="sleep-range" className="text-slate-300 font-semibold">
                    Nightly Sleep Duration
                  </label>
                  <span className="font-mono font-bold text-purple-300">{student.sleep_hours.toFixed(1)} hrs</span>
                </div>
                <input
                  id="sleep-range"
                  type="range"
                  min="4"
                  max="11"
                  step="0.5"
                  value={student.sleep_hours}
                  onChange={(e) => setStudent({ ...student, sleep_hours: parseFloat(e.target.value) })}
                  className="w-full cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <label htmlFor="tutoring-range" className="text-slate-300 font-semibold flex items-center gap-1">
                    <Users className="w-3.5 h-3.5 text-yellow-400" /> Tutoring Sessions
                  </label>
                  <span className="font-mono font-bold text-yellow-300">{student.tutoring_sessions} / mo</span>
                </div>
                <input
                  id="tutoring-range"
                  type="range"
                  min="0"
                  max="12"
                  step="1"
                  value={student.tutoring_sessions}
                  onChange={(e) => setStudent({ ...student, tutoring_sessions: parseInt(e.target.value, 10) })}
                  className="w-full cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <label htmlFor="stress-range" className="text-slate-300 font-semibold">
                    Stress Index (1-10)
                  </label>
                  <span className="font-mono font-bold text-rose-300">{student.stress_level} / 10</span>
                </div>
                <input
                  id="stress-range"
                  type="range"
                  min="1"
                  max="10"
                  step="1"
                  value={student.stress_level}
                  onChange={(e) => setStudent({ ...student, stress_level: parseInt(e.target.value, 10) })}
                  className="w-full cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <label htmlFor="screen-range" className="text-slate-300 font-semibold">
                    Recreational Screen Time
                  </label>
                  <span className="font-mono font-bold text-orange-300">{student.screen_time_hours.toFixed(1)} hrs/day</span>
                </div>
                <input
                  id="screen-range"
                  type="range"
                  min="0"
                  max="10"
                  step="0.5"
                  value={student.screen_time_hours}
                  onChange={(e) => setStudent({ ...student, screen_time_hours: parseFloat(e.target.value) })}
                  className="w-full cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <label htmlFor="physical-range" className="text-slate-300 font-semibold">
                    Physical Activity
                  </label>
                  <span className="font-mono font-bold text-emerald-300">{student.physical_activity_hours.toFixed(1)} hrs/wk</span>
                </div>
                <input
                  id="physical-range"
                  type="range"
                  min="0"
                  max="18"
                  step="0.5"
                  value={student.physical_activity_hours}
                  onChange={(e) => setStudent({ ...student, physical_activity_hours: parseFloat(e.target.value) })}
                  className="w-full cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <label htmlFor="commute-range" className="text-slate-300 font-semibold">
                    Daily Commute Time
                  </label>
                  <span className="font-mono font-bold text-slate-300">{student.commute_minutes} mins</span>
                </div>
                <input
                  id="commute-range"
                  type="range"
                  min="0"
                  max="120"
                  step="5"
                  value={student.commute_minutes}
                  onChange={(e) => setStudent({ ...student, commute_minutes: parseInt(e.target.value, 10) })}
                  className="w-full cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Panel 3: Support & Environment */}
          <div className="bg-edu-900 border border-edu-800 rounded-2xl p-5 shadow-xl space-y-4">
            <h3 className="text-xs font-extrabold uppercase tracking-wider text-slate-300 flex items-center gap-2 border-b border-edu-800 pb-2">
              <Sliders className="w-4 h-4 text-emerald-400" />
              3. Support Ecosystem & Context (8 Categorical Features)
            </h3>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
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
                <label className="text-slate-400 block mb-1">Motivation Level</label>
                <select
                  value={student.motivation_level}
                  onChange={(e) => setStudent({ ...student, motivation_level: e.target.value as any })}
                  className="w-full bg-edu-950 border border-edu-800 rounded-lg p-2 text-white outline-none focus:border-indigo-500"
                >
                  <option value="Low">Low</option>
                  <option value="Medium">Medium</option>
                  <option value="High">High</option>
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Resource Access</label>
                <select
                  value={student.access_to_resources}
                  onChange={(e) => setStudent({ ...student, access_to_resources: e.target.value as any })}
                  className="w-full bg-edu-950 border border-edu-800 rounded-lg p-2 text-white outline-none focus:border-indigo-500"
                >
                  <option value="Low">Low</option>
                  <option value="Medium">Medium</option>
                  <option value="High">High</option>
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Parent Education</label>
                <select
                  value={student.parental_education}
                  onChange={(e) => setStudent({ ...student, parental_education: e.target.value as any })}
                  className="w-full bg-edu-950 border border-edu-800 rounded-lg p-2 text-white outline-none focus:border-indigo-500"
                >
                  <option value="High School">High School</option>
                  <option value="Bachelor">Bachelor</option>
                  <option value="Master+">Master+</option>
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

              <div>
                <label className="text-slate-400 block mb-1">School Type</label>
                <select
                  value={student.school_type}
                  onChange={(e) => setStudent({ ...student, school_type: e.target.value as any })}
                  className="w-full bg-edu-950 border border-edu-800 rounded-lg p-2 text-white outline-none focus:border-indigo-500"
                >
                  <option value="Public">Public</option>
                  <option value="Private">Private</option>
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Learning Style</label>
                <select
                  value={student.learning_style}
                  onChange={(e) => setStudent({ ...student, learning_style: e.target.value as any })}
                  className="w-full bg-edu-950 border border-edu-800 rounded-lg p-2 text-white outline-none focus:border-indigo-500"
                >
                  <option value="Visual">Visual</option>
                  <option value="Auditory">Auditory</option>
                  <option value="Kinesthetic">Kinesthetic</option>
                  <option value="Reading/Writing">Reading/Writing</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Live Outcome, Engineered Signals & Factor Attribution (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-edu-900 border border-edu-800 rounded-2xl p-6 shadow-xl space-y-5 sticky top-24">
            <div className="flex items-center justify-between border-b border-edu-800 pb-3">
              <span className="text-xs uppercase font-extrabold tracking-wider text-slate-400">
                REAL-TIME PREDICTION
              </span>
              <span className="text-xs font-mono text-indigo-400">
                {loading ? 'INFERENCING...' : prediction?.model_used.toUpperCase()}
              </span>
            </div>

            {prediction && (
              <>
                <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-edu-950 border border-edu-800 rounded-2xl p-5">
                  <div>
                    <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold block">
                      Projected Final Score
                    </span>
                    <div className="text-4xl sm:text-5xl font-black text-white mt-1">
                      {prediction.predicted_score.toFixed(1)}{' '}
                      <span className="text-lg text-slate-500 font-normal">/ 100</span>
                    </div>
                    <div className="text-[11px] font-mono text-slate-400 mt-1">
                      80% Interval: [{prediction.confidence_interval[0].toFixed(1)} — {prediction.confidence_interval[1].toFixed(1)}]
                    </div>
                  </div>

                  <div className="text-right flex flex-col items-end">
                    <span
                      className={`px-4 py-1.5 rounded-full text-xs font-bold border flex items-center gap-1.5 ${getTierColor(
                        prediction.grade_tier
                      )}`}
                    >
                      <span>{prediction.status_icon}</span>
                      <span>{prediction.grade_tier}</span>
                    </span>
                    <span className="text-xs text-slate-400 mt-1.5 max-w-[190px] text-right font-medium">
                      {prediction.tier_label}
                    </span>
                  </div>
                </div>

                {/* Progress Indicator */}
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between text-slate-400">
                    <span>Mastery Gauge</span>
                    <span className="font-mono text-slate-200">{prediction.predicted_score.toFixed(1)}%</span>
                  </div>
                  <progress max={100} value={prediction.predicted_score} className="w-full h-2 rounded" />
                </div>

                {/* Secondary Indicators */}
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="bg-edu-950 border border-edu-800 rounded-xl p-3">
                    <span className="text-slate-400 text-[10px] uppercase font-bold tracking-wider block">
                      Delta vs Midterm
                    </span>
                    <div className="flex items-center gap-1 text-lg font-bold mt-0.5">
                      {prediction.delta_vs_previous >= 0 ? (
                        <>
                          <ArrowUpRight className="w-4 h-4 text-emerald-400" />
                          <span className="text-emerald-400">+{prediction.delta_vs_previous.toFixed(1)} pts</span>
                        </>
                      ) : (
                        <>
                          <ArrowDownRight className="w-4 h-4 text-rose-400" />
                          <span className="text-rose-400">{prediction.delta_vs_previous.toFixed(1)} pts</span>
                        </>
                      )}
                    </div>
                  </div>

                  <div className="bg-edu-950 border border-edu-800 rounded-xl p-3">
                    <span className="text-slate-400 text-[10px] uppercase font-bold tracking-wider block">
                      Engagement Score
                    </span>
                    <div className="text-lg font-bold text-indigo-300 mt-0.5 font-mono">
                      {(prediction.engineered_features.engagement_score * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Live Engineered Features Strip */}
                <div className="bg-edu-950 border border-edu-800 rounded-xl p-3.5 space-y-2 text-xs">
                  <div className="flex items-center gap-1.5 text-indigo-300 font-bold">
                    <Sparkles className="w-3.5 h-3.5" /> Live Engineered Signals (Leak-Free Pipeline)
                  </div>
                  <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
                    <div className="flex justify-between bg-edu-900/60 px-2.5 py-1.5 rounded">
                      <span className="text-slate-400">Momentum</span>
                      <span className="text-white">{prediction.engineered_features.academic_momentum?.toFixed(1)}</span>
                    </div>
                    <div className="flex justify-between bg-edu-900/60 px-2.5 py-1.5 rounded">
                      <span className="text-slate-400">Study Eff.</span>
                      <span className="text-white">{prediction.engineered_features.study_efficiency?.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between bg-edu-900/60 px-2.5 py-1.5 rounded">
                      <span className="text-slate-400">Sleep Deficit</span>
                      <span className="text-white">{prediction.engineered_features.sleep_deficit?.toFixed(2)}h</span>
                    </div>
                    <div className="flex justify-between bg-edu-900/60 px-2.5 py-1.5 rounded">
                      <span className="text-slate-400">Wellbeing</span>
                      <span className="text-white">{prediction.engineered_features.wellbeing_index?.toFixed(2)}</span>
                    </div>
                  </div>
                </div>

                {/* Local Per-Student Attribution */}
                {prediction.feature_contributions.length > 0 && (
                  <div className="bg-edu-950 border border-edu-800 rounded-xl p-3.5 space-y-2 text-xs">
                    <div className="flex items-center justify-between text-slate-300 font-bold">
                      <span className="flex items-center gap-1.5">
                        <Award className="w-3.5 h-3.5 text-indigo-400" /> Local Factor Attribution
                      </span>
                      <span className="text-[10px] font-mono text-slate-500">pts vs cohort avg</span>
                    </div>
                    <div className="space-y-1.5">
                      {prediction.feature_contributions.slice(0, 6).map((c) => {
                        const pct = Math.min(100, Math.abs(c.impact) * 8);
                        const pos = c.impact >= 0;
                        return (
                          <div key={c.feature} className="space-y-0.5">
                            <div className="flex justify-between text-[11px]">
                              <span className="text-slate-300">{c.label}</span>
                              <span className={`font-mono font-bold ${pos ? 'text-emerald-400' : 'text-rose-400'}`}>
                                {pos ? `+${c.impact.toFixed(2)}` : c.impact.toFixed(2)} pts
                              </span>
                            </div>
                            <div className="w-full h-1.5 bg-edu-900 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full ${pos ? 'bg-emerald-500' : 'bg-rose-500'}`}
                                style={{ width: `${Math.max(pct, 4)}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Advisory Recommendations */}
                <div className="bg-edu-950/90 border border-edu-800 rounded-xl p-4 space-y-2 text-xs">
                  <div className="flex items-center gap-1.5 text-indigo-400 font-bold">
                    <Lightbulb className="w-4 h-4" /> Real-Time Intervention Guidance
                  </div>
                  <div className="space-y-1.5 pt-1">
                    {prediction.behavioral_diagnostics.recommendations.map((rec, i) => (
                      <div key={i} className="flex items-start gap-2 text-slate-300">
                        <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 mt-1.5 flex-shrink-0" />
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
