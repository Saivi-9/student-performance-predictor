import React, { useState, useEffect } from 'react';
import { Navbar, type StudioView } from './components/Navbar';
import { StudentForecaster } from './components/StudentForecaster';
import { SimulationStudio } from './components/SimulationStudio';
import { ModelBenchmarks } from './components/ModelBenchmarks';
import { PretrainedHub } from './components/PretrainedHub';
import type { StudentInput, BenchmarkReport, HealthStatus } from './types';
import { fetchBenchmarks, fetchHealth } from './api/client';
import { Sparkles, ShieldCheck } from 'lucide-react';

const DEFAULT_STUDENT: StudentInput = {
  hours_studied: 12.0,
  attendance_percent: 88.0,
  previous_score: 74.0,
  quiz_average: 78.0,
  assignment_completion_rate: 85.0,
  practice_papers_solved: 5,
  sleep_hours: 7.5,
  tutoring_sessions: 2,
  physical_activity_hours: 4.5,
  screen_time_hours: 3.0,
  stress_level: 4,
  commute_minutes: 25,
  parental_involvement: 'Medium',
  motivation_level: 'Medium',
  access_to_resources: 'Medium',
  parental_education: 'Bachelor',
  extracurricular: 'Yes',
  internet_access: 'Yes',
  school_type: 'Public',
  learning_style: 'Visual',
};

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<StudioView>('forecast');
  const [student, setStudent] = useState<StudentInput>(DEFAULT_STUDENT);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [benchmarks, setBenchmarks] = useState<BenchmarkReport | null>(null);
  const [benchLoading, setBenchLoading] = useState<boolean>(true);
  const [modelSlug, setModelSlug] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth().then(setHealth);
    fetchBenchmarks()
      .then(setBenchmarks)
      .finally(() => setBenchLoading(false));
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} health={health} />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-8">
        {activeTab === 'forecast' && (
          <StudentForecaster
            student={student}
            setStudent={setStudent}
            models={benchmarks?.models || []}
            modelSlug={modelSlug}
            setModelSlug={setModelSlug}
          />
        )}

        {activeTab === 'simulator' && (
          <SimulationStudio
            student={student}
            models={benchmarks?.models || []}
            modelSlug={modelSlug}
            setModelSlug={setModelSlug}
          />
        )}

        {activeTab === 'explainability' && (
          <ModelBenchmarks
            benchmarks={benchmarks}
            loading={benchLoading}
            activeModelSlug={modelSlug}
            onSelectModel={(slug) => {
              setModelSlug(slug);
              setActiveTab('forecast');
            }}
          />
        )}

        {activeTab === 'hub' && <PretrainedHub student={student} modelSlug={modelSlug} />}
      </main>

      <footer className="border-t border-slate-900 bg-slate-950/80 backdrop-blur-md px-6 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
            </span>
            <span>Academic Performance Studio v2.0 • FastAPI + LightGBM / Stacking / Hugging Face Hub</span>
          </div>

          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={() => setStudent(DEFAULT_STUDENT)}
              className="hover:text-slate-300 transition underline underline-offset-4"
            >
              Reset Student Profile
            </button>
            <span className="text-slate-700">•</span>
            <span className="flex items-center gap-1 text-slate-500">
              <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
              {benchmarks
                ? `Champion R² ${benchmarks.test_metrics.R2.toFixed(4)} (${benchmarks.champion_model})`
                : 'Awaiting Benchmarks'}
            </span>
            <span className="text-slate-700">•</span>
            <span className="flex items-center gap-1 text-slate-500">
              Built with <Sparkles className="w-3 h-3 text-indigo-400" /> for Real-Time Academic ML
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
