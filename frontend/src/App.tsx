import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { StudentForecaster } from './components/StudentForecaster';
import { SimulationStudio } from './components/SimulationStudio';
import { ModelBenchmarks } from './components/ModelBenchmarks';
import type { StudentInput } from './types';
import { Sparkles, ShieldCheck } from 'lucide-react';

const DEFAULT_STUDENT: StudentInput = {
  hours_studied: 12.0,
  attendance_percent: 88.0,
  previous_score: 74.0,
  sleep_hours: 7.5,
  tutoring_sessions: 2,
  parental_involvement: 'Medium',
  extracurricular: 'Yes',
  internet_access: 'Yes',
};

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'forecast' | 'simulator' | 'explainability'>('forecast');
  const [student, setStudent] = useState<StudentInput>(DEFAULT_STUDENT);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-8">
        {activeTab === 'forecast' && (
          <StudentForecaster student={student} setStudent={setStudent} />
        )}

        {activeTab === 'simulator' && (
          <SimulationStudio student={student} />
        )}

        {activeTab === 'explainability' && (
          <ModelBenchmarks />
        )}
      </main>

      <footer className="border-t border-slate-900 bg-slate-950/80 backdrop-blur-md px-6 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span>Academic Performance Studio • FastAPI + Scikit-Learn L2 Engine</span>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={() => setStudent(DEFAULT_STUDENT)}
              className="hover:text-slate-300 transition underline underline-offset-4"
            >
              Reset to Default Student
            </button>
            <span className="text-slate-700">•</span>
            <span className="flex items-center gap-1 text-slate-500">
              <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" /> Model Calibrated (R² 0.8385)
            </span>
            <span className="text-slate-700">•</span>
            <span className="flex items-center gap-1 text-slate-500">
              Built with <Sparkles className="w-3 h-3 text-indigo-400" /> for Academic Coaching
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
