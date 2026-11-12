import React from 'react';
import { GraduationCap, Award, Cpu, Sparkles, Globe } from 'lucide-react';
import type { HealthStatus } from '../types';

export type StudioView = 'forecast' | 'simulator' | 'explainability' | 'hub';

interface NavbarProps {
  activeTab: StudioView;
  setActiveTab: (tab: StudioView) => void;
  health: HealthStatus | null;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, health }) => {
  const navItems: { id: StudioView; label: string; icon: React.ReactNode }[] = [
    { id: 'forecast', label: 'Live Forecaster', icon: <Sparkles className="w-3.5 h-3.5" /> },
    { id: 'simulator', label: 'What-If Simulator', icon: <Cpu className="w-3.5 h-3.5" /> },
    { id: 'explainability', label: 'Model Benchmarks', icon: <Award className="w-3.5 h-3.5" /> },
    { id: 'hub', label: 'Data & HF Hub', icon: <Globe className="w-3.5 h-3.5" /> },
  ];

  return (
    <header className="border-b border-edu-800 bg-edu-900/90 backdrop-blur-md sticky top-0 z-50 px-6 py-3.5">
      <div className="max-w-7xl mx-auto flex flex-col lg:flex-row items-center justify-between gap-4">
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
                v{health?.version || '2.0.0'} • {health?.total_features || 28} FEATURES
              </span>
            </div>
            <p className="text-xs text-slate-400">
              {health ? `${health.dataset_rows.toLocaleString()} Student Records • ${health.models_available} Benchmarked Models` : 'Multi-Model Student Performance & HF Pre-Trained Hub'}
            </p>
          </div>
        </div>

        {/* Single-row Horizontal Page Navigation */}
        <nav
          aria-label="Primary studio navigation"
          className="flex items-center flex-nowrap overflow-x-auto gap-1.5 bg-edu-950/80 p-1 rounded-xl border border-edu-800 max-w-full"
        >
          {navItems.map((item) => {
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                aria-current={active ? 'page' : undefined}
                onClick={() => setActiveTab(item.id)}
                className={`relative flex-none flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition whitespace-nowrap ${
                  active
                    ? 'bg-edu-800 text-indigo-300 shadow-sm border border-indigo-500/30'
                    : 'text-slate-400 hover:text-white hover:bg-edu-850'
                }`}
              >
                {item.icon}
                <span>{item.label}</span>
                {active && (
                  <span
                    aria-hidden="true"
                    className="absolute bottom-0 left-2 right-2 h-0.5 bg-indigo-400 rounded-full"
                  />
                )}
              </button>
            );
          })}
        </nav>

        {/* Live Backend Status */}
        <div className="flex items-center gap-2 font-mono text-xs text-slate-300">
          <span
            className={`w-2 h-2 rounded-full ${
              health?.model_loaded ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
            }`}
          />
          <span className="truncate max-w-[240px]">
            {health?.model_loaded
              ? `CHAMPION: ${health.champion_model.toUpperCase()}`
              : 'CONNECTING / OFFLINE FALLBACK'}
          </span>
        </div>
      </div>
    </header>
  );
};
