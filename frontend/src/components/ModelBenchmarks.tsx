import React from "react";
import { CheckCircle2, BarChart2, ShieldCheck, TrendingUp, Layers, Award, Cpu, AlertCircle } from "lucide-react";

export const ModelBenchmarks: React.FC = () => {
  const metrics = [
    { label: "Test R² Score", value: "0.8385", sub: "83.85% variance explained", badge: "Best In Class", color: "text-emerald-400" },
    { label: "RMSE", value: "2.984 pts", sub: "Residual spread on 0-100 scale", badge: "Low Error", color: "text-indigo-400" },
    { label: "MAE", value: "2.366 pts", sub: "Average absolute point deviation", badge: "Calibrated", color: "text-blue-400" },
    { label: "Dataset Cohort", value: "10,000", sub: "5-fold stratified cross-validation", badge: "Verified", color: "text-amber-400" },
  ];

  const modelComparisons = [
    { name: "Ridge Regressor (L2 Optimized)", r2: "0.8385", rmse: "2.984", status: "Production Champion", selected: true },
    { name: "Ordinary Least Squares (OLS)", r2: "0.8385", rmse: "2.985", status: "Baseline Equivalent", selected: false },
    { name: "Gradient Boosting Regressor", r2: "0.8123", rmse: "3.210", status: "Non-Linear Candidate", selected: false },
    { name: "Random Forest Regressor", r2: "0.8012", rmse: "3.315", status: "Ensemble Baseline", selected: false },
  ];

  const featureImportances = [
    { name: "Previous Exam Score (previous_score)", score: 0.721, pct: 72.1, color: "bg-indigo-500", desc: "Prior mastery anchor" },
    { name: "Weekly Study Time (hours_studied)", score: 0.185, pct: 18.5, color: "bg-blue-500", desc: "Primary actionable effort lever" },
    { name: "Classroom Attendance (attendance_percent)", score: 0.054, pct: 5.4, color: "bg-emerald-500", desc: "Syllabus continuity & immersion" },
    { name: "Tutoring Sessions (tutoring_sessions)", score: 0.021, pct: 2.1, color: "bg-teal-500", desc: "1-on-1 concept remediation" },
    { name: "Nightly Sleep (sleep_hours)", score: 0.012, pct: 1.2, color: "bg-amber-500", desc: "Cognitive consolidation limit" },
    { name: "Extracurriculars (extracurricular_activities)", score: 0.007, pct: 0.7, color: "bg-purple-500", desc: "Holistic student enrichment" },
    { name: "Physical Activity (physical_activity_hours)", score: 0.004, pct: 0.4, color: "bg-rose-500", desc: "Stress buffering & focus" },
  ];

  return (
    <div className="space-y-6">
      {/* Title & Introduction */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 backdrop-blur-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-6 h-6 text-indigo-400" />
              <h2 className="text-xl font-bold text-white tracking-wide">Model Validation & Benchmark Dossier</h2>
            </div>
            <p className="text-sm text-slate-400 mt-1">
              Rigorous empirical evaluation protocol based on 10,000 multi-variable academic student records.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-3 py-1 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold rounded-full flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" /> Generalization Verified
            </span>
            <span className="px-3 py-1 bg-slate-800 border border-slate-700 text-slate-300 text-xs font-mono rounded-full">
              v1.2.0 Model Artifact
            </span>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
          {metrics.map((m, idx) => (
            <div key={idx} className="bg-slate-950/70 border border-slate-800/80 rounded-lg p-4 relative overflow-hidden">
              <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
                <span>{m.label}</span>
                <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 bg-slate-800 rounded text-slate-300">
                  {m.badge}
                </span>
              </div>
              <div className={`text-2xl font-black mt-2 font-mono ${m.color}`}>{m.value}</div>
              <div className="text-xs text-slate-500 mt-1">{m.sub}</div>
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-indigo-500/40 to-transparent"></div>
            </div>
          ))}
        </div>
      </div>

      {/* Permutation Importance & Model Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Permutation Feature Importance */}
        <div className="lg:col-span-7 bg-slate-900/60 border border-slate-800 rounded-xl p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <BarChart2 className="w-5 h-5 text-indigo-400" />
              <h3 className="text-base font-semibold text-white">Permutation Feature Importance</h3>
            </div>
            <span className="text-xs text-slate-500 font-mono">Mean Dropout Δ R²</span>
          </div>
          <p className="text-xs text-slate-400 mb-6">
            Evaluates the reduction in model predictive accuracy when shuffling each individual input feature across test folds.
          </p>

          <div className="space-y-4">
            {featureImportances.map((f, i) => (
              <div key={i} className="space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-300 font-medium">{f.name}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-slate-500 text-[11px]">{f.desc}</span>
                    <span className="font-mono font-semibold text-indigo-300 w-12 text-right">
                      {(f.score * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div className="w-full h-2 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className={`h-full ${f.color} rounded-full transition-all duration-500`}
                    style={{ width: `${Math.max(f.pct, 1.5)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 pt-4 border-t border-slate-800 text-xs text-slate-400 flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <span>
              <strong>Key Finding:</strong> Prior score provides the initial anchor baseline (~72%), while weekly study hours and attendance provide the highest modifiable leverage (~24%) for shifting a student across grade boundaries.
            </span>
          </div>
        </div>

        {/* Model Benchmark Table */}
        <div className="lg:col-span-5 bg-slate-900/60 border border-slate-800 rounded-xl p-6 backdrop-blur-sm flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Award className="w-5 h-5 text-indigo-400" />
              <h3 className="text-base font-semibold text-white">Algorithm Tournament Comparison</h3>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Evaluated on identical 20% holdout test partitions using scikit-learn standard scaling.
            </p>

            <div className="space-y-3">
              {modelComparisons.map((c, i) => (
                <div
                  key={i}
                  className={`p-3.5 rounded-lg border transition-all ${
                    c.selected
                      ? "bg-indigo-950/30 border-indigo-500/50 ring-1 ring-indigo-500/30"
                      : "bg-slate-950/60 border-slate-800/80"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className={`text-xs font-semibold ${c.selected ? "text-indigo-200" : "text-slate-300"}`}>
                      {c.name}
                    </span>
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                        c.selected
                          ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40"
                          : "bg-slate-800 text-slate-400"
                      }`}
                    >
                      {c.status}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 mt-2 pt-2 border-t border-slate-800/60 text-xs font-mono">
                    <div>
                      <span className="text-slate-500">R² Score: </span>
                      <span className="text-slate-200 font-semibold">{c.r2}</span>
                    </div>
                    <div>
                      <span className="text-slate-500">RMSE: </span>
                      <span className="text-slate-200 font-semibold">{c.rmse} pts</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-slate-800 flex items-center justify-between text-xs text-slate-500">
            <span className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-slate-400" />
              Engine: Scikit-Learn 1.6
            </span>
            <span className="flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-slate-400" />
              Pipeline: StandardScaler + Regressor
            </span>
          </div>
        </div>
      </div>

      {/* Methodological Transparency Statement */}
      <div className="bg-slate-900/40 border border-slate-800/80 rounded-xl p-5 text-xs text-slate-400 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <TrendingUp className="w-5 h-5 text-indigo-400 shrink-0" />
          <div>
            <div className="font-semibold text-slate-200">Ethical & Methodological Transparency Note</div>
            <p className="text-slate-400 mt-0.5">
              Predictions are probabilistic forecasts designed for educational coaching and early intervention rather than punitive decision-making.
            </p>
          </div>
        </div>
        <div className="shrink-0 text-slate-500 font-mono text-[11px]">
          Audited under FERPA & Algorithmic Fairness Guidelines
        </div>
      </div>
    </div>
  );
};
