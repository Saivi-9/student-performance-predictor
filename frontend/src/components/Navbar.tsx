import React from 'react';
import { GraduationCap, Award, Cpu, Sparkles } from 'lucide-react';

interface NavbarProps {
  activeTab: 'forecast' | 'simulator' | 'explainability';
  setActiveTab: (tab: 'forecast' | 'simulator' | 'explainability') => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab }) => {
  return (
    <header className="border-b border-edu-800 bg-edu-900/90 backdrop-blur-md sticky top-0 z-50 px-6 py-3.5">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-600/30 border border-indigo-500/40 flex items-center justify-center text-indigo-400 shadow-lg shadow-indigo-500/10">
            <GraduationCap className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-indigo-300 bg-clip-text text-transparent">
                ACADEMIC AI STUDIO
              </span>
              <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
                SCIKIT-LEARN v1.3
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Student Performance Forecasting & Intervention Analytics
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1.5 bg-edu-950/80 p-1 rounded-xl border border-edu-800">
          <button
            onClick={() => setActiveTab('forecast')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
              activeTab === 'forecast'
                ? 'bg-edu-800 text-indigo-400 shadow-sm border border-indigo-500/30'
                : 'text-slate-400 hover:text-white hover:bg-edu-850'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            Live Forecaster
          </button>

          <button
            onClick={() => setActiveTab('simulator')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
              activeTab === 'simulator'
                ? 'bg-edu-800 text-indigo-400 shadow-sm border border-indigo-500/30'
                : 'text-slate-400 hover:text-white hover:bg-edu-850'
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            'What-If' Simulator
          </button>

          <button
            onClick={() => setActiveTab('explainability')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
              activeTab === 'explainability'
                ? 'bg-edu-800 text-indigo-400 shadow-sm border border-indigo-500/30'
                : 'text-slate-400 hover:text-white hover:bg-edu-850'
            }`}
          >
            <Award className="w-3.5 h-3.5" />
            Model Benchmarks
          </button>
        </nav>

        {/* Status indicator */}
        <div className="flex items-center gap-2 font-mono text-xs text-slate-300">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>CHAMPION: RIDGE PIPELINE (R² = 0.8385)</span>
        </div>
      </div>
    </header>
  );
};
