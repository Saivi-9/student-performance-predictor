import React, { useState, useEffect } from 'react';
import {
  Search,
  Globe,
  Database,
  Play,
  Users,
  ExternalLink,
  Loader2,
  CheckCircle2,
  Layers,
} from 'lucide-react';
import type {
  StudentInput,
  PretrainedCatalogEntry,
  HubSearchItem,
  PretrainedBenchmarkEntry,
  BatchPredictionResponse,
  DatasetSummaryResponse,
} from '../types';
import {
  fetchPretrainedCatalog,
  searchPretrainedHub,
  runPretrainedBenchmark,
  runPretrainedPrediction,
  fetchDatasetSummary,
  fetchBatchPrediction,
  fetchPrediction,
} from '../api/client';

interface PretrainedHubProps {
  student: StudentInput;
  modelSlug: string | null;
}

type HubSubTab = 'hub' | 'cohort' | 'batch';

export const PretrainedHub: React.FC<PretrainedHubProps> = ({ student, modelSlug }) => {
  const [subTab, setSubTab] = useState<HubSubTab>('hub');

  // HF Pretrained state
  const [catalog, setCatalog] = useState<PretrainedCatalogEntry[]>([]);
  const [searchQuery, setSearchQuery] = useState('student performance');
  const [searchKind, setSearchKind] = useState<'models' | 'datasets'>('models');
  const [searchResults, setSearchResults] = useState<HubSearchItem[]>([]);
  const [searching, setSearching] = useState(false);
  const [pretrainedBenchmarks, setPretrainedBenchmarks] = useState<PretrainedBenchmarkEntry[]>([]);
  const [benchLoading, setBenchLoading] = useState(false);
  const [comparisonResult, setComparisonResult] = useState<{
    localScore: number;
    localModel: string;
    hfScore: number;
    hfRepo: string;
  } | null>(null);
  const [comparing, setComparing] = useState(false);

  // Cohort dataset explorer state
  const [datasetInfo, setDatasetInfo] = useState<DatasetSummaryResponse | null>(null);

  // Batch scoring state
  const [batchSize, setBatchSize] = useState(8);
  const [batchResult, setBatchResult] = useState<BatchPredictionResponse | null>(null);
  const [batchLoading, setBatchLoading] = useState(false);

  useEffect(() => {
    fetchPretrainedCatalog().then(setCatalog);
    fetchDatasetSummary(12).then(setDatasetInfo);
  }, []);

  const handleHubSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setSearching(true);
    try {
      const res = await searchPretrainedHub(searchQuery, searchKind, 10);
      setSearchResults(res);
    } finally {
      setSearching(false);
    }
  };

  const handleBenchmarkPretrained = async () => {
    setBenchLoading(true);
    try {
      const res = await runPretrainedBenchmark(false);
      setPretrainedBenchmarks(res);
    } finally {
      setBenchLoading(false);
    }
  };

  const handleCompareLive = async () => {
    setComparing(true);
    try {
      const [localPred, hfPred] = await Promise.all([
        fetchPrediction(student, modelSlug),
        runPretrainedPrediction(student),
      ]);
      setComparisonResult({
        localScore: localPred.predicted_score,
        localModel: localPred.model_used,
        hfScore: hfPred.predicted_score,
        hfRepo: hfPred.repo_id,
      });
    } finally {
      setComparing(false);
    }
  };

  const handleRunSyntheticBatch = async () => {
    setBatchLoading(true);
    try {
      const generated: StudentInput[] = Array.from({ length: batchSize }, (_, idx) => {
        const offset = (idx - batchSize / 2) * 3.5;
        return {
          ...student,
          previous_score: Math.max(35, Math.min(99, Math.round(student.previous_score + offset))),
          quiz_average: Math.max(35, Math.min(99, Math.round(student.quiz_average + offset * 0.9))),
          hours_studied: Math.max(2, Math.min(28, Number((student.hours_studied + idx * 0.8 - 3).toFixed(1)))),
          attendance_percent: Math.max(50, Math.min(100, Math.round(student.attendance_percent + offset * 0.5))),
        };
      });
      const res = await fetchBatchPrediction(generated, modelSlug);
      setBatchResult(res);
    } finally {
      setBatchLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Sub-navigation using WAI-ARIA tabs pattern inside a single horizontal row */}
      <div className="bg-edu-900 border border-edu-800 rounded-2xl p-4 shadow-xl flex flex-wrap items-center justify-between gap-4">
        <div
          role="tablist"
          aria-label="Data Hub and Batch Workflows"
          className="bg-edu-950 p-1 rounded-xl border border-edu-800"
        >
          <button
            type="button"
            role="tab"
            id="tab-hf-hub"
            aria-selected={subTab === 'hub'}
            onClick={() => setSubTab('hub')}
            className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition ${
              subTab === 'hub' ? 'bg-edu-800 text-indigo-300' : 'text-slate-400 hover:text-white'
            }`}
          >
            <Globe className="w-3.5 h-3.5" /> Hugging Face Pre-Trained Hub
          </button>

          <button
            type="button"
            role="tab"
            id="tab-cohort-data"
            aria-selected={subTab === 'cohort'}
            onClick={() => setSubTab('cohort')}
            className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition ${
              subTab === 'cohort' ? 'bg-edu-800 text-indigo-300' : 'text-slate-400 hover:text-white'
            }`}
          >
            <Database className="w-3.5 h-3.5" /> Multi-Source Cohort Explorer
          </button>

          <button
            type="button"
            role="tab"
            id="tab-batch-score"
            aria-selected={subTab === 'batch'}
            onClick={() => setSubTab('batch')}
            className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition ${
              subTab === 'batch' ? 'bg-edu-800 text-indigo-300' : 'text-slate-400 hover:text-white'
            }`}
          >
            <Users className="w-3.5 h-3.5" /> Real-Time Batch Scoring
          </button>
        </div>

        <div className="text-xs font-mono text-slate-400">
          {datasetInfo?.external_sources.length || 0} External Real Datasets Cached
        </div>
      </div>

      {/* Panel 1: Hugging Face Hub */}
      <div
        role="tabpanel"
        aria-labelledby="tab-hf-hub"
        hidden={subTab !== 'hub' ? ('until-found' as any) : false}
        tabIndex={subTab === 'hub' ? 0 : undefined}
      >
        <div className="space-y-6">
          {/* Live Search + Quick Inference Comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-7 bg-edu-900 border border-edu-800 rounded-2xl p-6 space-y-4">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Search className="w-4 h-4 text-indigo-400" />
                Live Hugging Face Hub Discovery (Models & Datasets)
              </h3>
              <form onSubmit={handleHubSearch} className="flex flex-wrap gap-2 text-xs">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search Hugging Face Hub (e.g., student performance)..."
                  className="flex-1 min-w-[200px] bg-edu-950 border border-edu-800 rounded-lg px-3 py-2 text-white outline-none focus:border-indigo-500"
                />
                <select
                  aria-label="Resource Kind"
                  value={searchKind}
                  onChange={(e) => setSearchKind(e.target.value as any)}
                  className="bg-edu-950 border border-edu-800 rounded-lg px-3 py-2 text-white"
                >
                  <option value="models">Models</option>
                  <option value="datasets">Datasets</option>
                </select>
                <button
                  type="submit"
                  disabled={searching}
                  className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold flex items-center gap-1.5 transition"
                >
                  {searching ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
                  Search Hub
                </button>
              </form>

              {searchResults.length > 0 && (
                <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                  {searchResults.map((item) => (
                    <div
                      key={item.id}
                      className="bg-edu-950 border border-edu-800 rounded-xl p-3 flex items-center justify-between gap-3 text-xs"
                    >
                      <div className="min-w-0">
                        <a
                          href={item.url}
                          target="_blank"
                          rel="noreferrer"
                          className="font-mono font-bold text-indigo-300 hover:underline flex items-center gap-1 truncate"
                        >
                          {item.id} <ExternalLink className="w-3 h-3 shrink-0" />
                        </a>
                        <div className="text-[11px] text-slate-400 mt-0.5 flex flex-wrap gap-2">
                          <span>Downloads: {item.downloads.toLocaleString()}</span>
                          <span>Likes: {item.likes}</span>
                          {item.pipeline_tag && <span className="text-slate-500">[{item.pipeline_tag}]</span>}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Live Pretrained vs Local Champion Comparison */}
            <div className="lg:col-span-5 bg-edu-900 border border-edu-800 rounded-2xl p-6 space-y-4">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Layers className="w-4 h-4 text-emerald-400" />
                Live Pre-Trained vs. Local Engine Check
              </h3>
              <p className="text-xs text-slate-400">
                Compare your current student profile side-by-side against the community{' '}
                <code className="text-indigo-300">Shalan1/student-performance-predictor-models</code> Random Forest downloaded from Hugging Face Hub.
              </p>

              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={handleCompareLive}
                  disabled={comparing}
                  className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold flex items-center gap-1.5 transition"
                >
                  {comparing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                  Compare Live Prediction
                </button>
                <button
                  type="button"
                  onClick={handleBenchmarkPretrained}
                  disabled={benchLoading}
                  className="px-4 py-2 rounded-lg bg-edu-800 hover:bg-edu-700 border border-edu-700 text-slate-200 text-xs font-semibold flex items-center gap-1.5 transition"
                >
                  {benchLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
                  Benchmark HF Models
                </button>
              </div>

              {comparisonResult && (
                <div className="grid grid-cols-2 gap-3 pt-2 text-xs">
                  <div className="bg-edu-950 border border-indigo-500/40 rounded-xl p-3">
                    <div className="text-[10px] font-mono text-indigo-300 uppercase">Local v2 Model</div>
                    <div className="text-2xl font-black text-white mt-1">
                      {comparisonResult.localScore.toFixed(1)}
                    </div>
                    <div className="text-[11px] text-slate-400 truncate">{comparisonResult.localModel}</div>
                  </div>
                  <div className="bg-edu-950 border border-emerald-500/40 rounded-xl p-3">
                    <div className="text-[10px] font-mono text-emerald-300 uppercase">HF Shalan1 RF</div>
                    <div className="text-2xl font-black text-white mt-1">
                      {comparisonResult.hfScore.toFixed(1)}
                    </div>
                    <div className="text-[11px] text-slate-400 truncate">{comparisonResult.hfRepo}</div>
                  </div>
                </div>
              )}

              {pretrainedBenchmarks.length > 0 && (
                <div className="space-y-2 pt-2 text-xs">
                  {pretrainedBenchmarks.map((b) => (
                    <div key={b.repo_id} className="bg-edu-950 border border-edu-800 rounded-xl p-3">
                      <div className="font-mono font-bold text-indigo-300">{b.repo_id}</div>
                      {b.error ? (
                        <div className="text-rose-400">{b.error}</div>
                      ) : (
                        <div className="text-slate-300 mt-1 font-mono text-[11px]">
                          {b.benchmark_dataset} • R²: <strong className="text-emerald-400">{b.r2}</strong> • RMSE: {b.rmse} ({b.target_scale})
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Curated Pretrained Catalog Table */}
          <div className="bg-edu-900 border border-edu-800 rounded-2xl p-6 space-y-4">
            <h3 id="pretrained-catalog-caption" className="text-sm font-bold text-white">
              Verified Hugging Face Community Models Catalog ({catalog.length} Models)
            </h3>
            <div tabIndex={0} role="region" aria-labelledby="pretrained-catalog-caption" className="table-wrapper">
              <table className="responsive-table text-xs">
                <caption className="sr-only">Curated Hugging Face Student Performance Pretrained Models</caption>
                <thead className="text-slate-300 uppercase text-[11px] font-mono border-b border-slate-800">
                  <tr>
                    <th className="py-3 px-4">Repository ID</th>
                    <th className="py-3 px-4">Framework</th>
                    <th className="py-3 px-4">Task</th>
                    <th className="py-3 px-4">Training Dataset</th>
                    <th className="py-3 px-4">Live Inference</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70">
                  {catalog.map((c) => (
                    <tr key={c.repo_id} className="hover:bg-slate-900/60">
                      <th className="py-3 px-4 font-mono text-indigo-300 whitespace-nowrap">
                        <a href={c.url} target="_blank" rel="noreferrer" className="hover:underline inline-flex items-center gap-1">
                          {c.repo_id} <ExternalLink className="w-3 h-3" />
                        </a>
                      </th>
                      <td data-label="Framework" className="py-3 px-4 text-slate-200">{c.framework}</td>
                      <td data-label="Task" className="py-3 px-4 text-slate-300">{c.task}</td>
                      <td data-label="Dataset" className="py-3 px-4 text-slate-400">{c.dataset}</td>
                      <td data-label="Live Inference" className="py-3 px-4">
                        <span
                          className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                            c.supports_live_inference
                              ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'
                              : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          {c.supports_live_inference ? 'ENABLED' : 'CATALOG'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Panel 2: Multi-Source Cohort Explorer */}
      <div
        role="tabpanel"
        aria-labelledby="tab-cohort-data"
        hidden={subTab !== 'cohort' ? ('until-found' as any) : false}
        tabIndex={subTab === 'cohort' ? 0 : undefined}
      >
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-edu-900 border border-edu-800 rounded-2xl p-5">
              <div className="text-xs font-mono text-indigo-300 uppercase">Primary Training Cohort</div>
              <div className="text-3xl font-black text-white mt-1">
                {datasetInfo?.dataset_summary?.total_rows?.toLocaleString() || '20,000'} rows
              </div>
              <div className="text-xs text-slate-400 mt-1">
                20 raw + 8 engineered features • Imputed real-world missingness
              </div>
            </div>
            {datasetInfo?.external_sources.map((src) => (
              <div key={src.name} className="bg-edu-900 border border-edu-800 rounded-2xl p-5">
                <div className="text-xs font-mono text-emerald-300 uppercase">Real External Benchmark</div>
                <div className="text-3xl font-black text-white mt-1">{src.rows.toLocaleString()} rows</div>
                <div className="text-xs text-slate-300 font-semibold mt-1">{src.name}</div>
                <div className="text-[11px] text-slate-500">{src.source} ({src.columns} cols)</div>
              </div>
            ))}
          </div>

          {datasetInfo && datasetInfo.sample_records.length > 0 && (
            <div className="bg-edu-900 border border-edu-800 rounded-2xl p-6 space-y-4">
              <h3 id="cohort-sample-caption" className="text-sm font-bold text-white">
                Primary Cohort Sample Telemetry (First 12 Records)
              </h3>
              <div tabIndex={0} role="region" aria-labelledby="cohort-sample-caption" className="table-wrapper">
                <table className="responsive-table text-xs font-mono">
                  <caption className="sr-only">Sample Student Records from Primary 20k Training Cohort</caption>
                  <thead className="text-slate-300 uppercase text-[10px] border-b border-slate-800">
                    <tr>
                      <th className="py-2.5 px-3">#</th>
                      <th className="py-2.5 px-3">Prior Score</th>
                      <th className="py-2.5 px-3">Quiz Avg</th>
                      <th className="py-2.5 px-3">Study Hrs</th>
                      <th className="py-2.5 px-3">Attend %</th>
                      <th className="py-2.5 px-3">Sleep Hrs</th>
                      <th className="py-2.5 px-3">Stress</th>
                      <th className="py-2.5 px-3">Motivation</th>
                      <th className="py-2.5 px-3 text-emerald-300">Final Score</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    {datasetInfo.sample_records.map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-900/60">
                        <th className="py-2.5 px-3 text-slate-400">Row {idx + 1}</th>
                        <td data-label="Prior Score" className="py-2.5 px-3">{row.previous_score}</td>
                        <td data-label="Quiz Avg" className="py-2.5 px-3">{row.quiz_average}</td>
                        <td data-label="Study Hrs" className="py-2.5 px-3">{row.hours_studied}</td>
                        <td data-label="Attend %" className="py-2.5 px-3">{row.attendance_percent}%</td>
                        <td data-label="Sleep Hrs" className="py-2.5 px-3">{row.sleep_hours}</td>
                        <td data-label="Stress" className="py-2.5 px-3">{row.stress_level}/10</td>
                        <td data-label="Motivation" className="py-2.5 px-3">{row.motivation_level}</td>
                        <td data-label="Final Score" className="py-2.5 px-3 font-bold text-emerald-400">{row.final_score}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Panel 3: Real-Time Batch Cohort Scoring */}
      <div
        role="tabpanel"
        aria-labelledby="tab-batch-score"
        hidden={subTab !== 'batch' ? ('until-found' as any) : false}
        tabIndex={subTab === 'batch' ? 0 : undefined}
      >
        <div className="bg-edu-900 border border-edu-800 rounded-2xl p-6 space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h3 className="text-sm font-bold text-white">Real-Time Batch Cohort Scoring Simulator</h3>
              <p className="text-xs text-slate-400">
                Generates a synthetic classroom sweep centered around your active student profile and scores the entire cohort via{' '}
                <code className="text-indigo-300">/v1/predict/batch</code>.
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <label htmlFor="batch-size-input" className="text-slate-300 font-mono">
                Cohort Size:
              </label>
              <input
                id="batch-size-input"
                type="number"
                min={3}
                max={30}
                value={batchSize}
                onChange={(e) => setBatchSize(Math.max(3, Math.min(30, parseInt(e.target.value || '8', 10))))}
                className="w-20 bg-edu-950 border border-edu-800 rounded-lg px-2.5 py-1.5 text-white font-mono"
              />
              <button
                type="button"
                onClick={handleRunSyntheticBatch}
                disabled={batchLoading}
                className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold flex items-center gap-1.5 transition"
              >
                {batchLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                Score Cohort Now
              </button>
            </div>
          </div>

          {batchResult && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
                <div className="bg-edu-950 border border-edu-800 rounded-xl p-4">
                  <span className="text-slate-400 uppercase font-mono text-[10px]">Batch Size / Engine</span>
                  <div className="text-xl font-black text-white mt-1">{batchResult.count} Students</div>
                  <div className="text-indigo-300 font-mono text-[11px]">{batchResult.model_used}</div>
                </div>
                <div className="bg-edu-950 border border-edu-800 rounded-xl p-4">
                  <span className="text-slate-400 uppercase font-mono text-[10px]">Cohort Mean Score</span>
                  <div className="text-xl font-black text-emerald-400 mt-1">{batchResult.average_score} / 100</div>
                </div>
                <div className="bg-edu-950 border border-edu-800 rounded-xl p-4">
                  <span className="text-slate-400 uppercase font-mono text-[10px]">At-Risk Flagged (&lt; 65)</span>
                  <div className="text-xl font-black text-rose-400 mt-1">{batchResult.at_risk_count} Students</div>
                </div>
              </div>

              <div tabIndex={0} role="region" aria-label="Batch Cohort Scored Output" className="table-wrapper">
                <table className="responsive-table text-xs font-mono">
                  <caption className="sr-only">Scored Student Batch Predictions</caption>
                  <thead className="text-slate-300 uppercase text-[10px] border-b border-slate-800">
                    <tr>
                      <th className="py-2.5 px-4">Student #</th>
                      <th className="py-2.5 px-4">Predicted Score</th>
                      <th className="py-2.5 px-4">Grade Tier</th>
                      <th className="py-2.5 px-4">Academic Momentum</th>
                      <th className="py-2.5 px-4">Engagement Score</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    {batchResult.predictions.map((p) => (
                      <tr key={p.index} className="hover:bg-slate-900/60">
                        <th className="py-2.5 px-4 text-slate-300">Student #{p.index}</th>
                        <td data-label="Predicted Score" className="py-2.5 px-4 font-bold text-white">
                          {p.predicted_score.toFixed(1)}
                        </td>
                        <td data-label="Grade Tier" className="py-2.5 px-4 text-indigo-300">
                          {p.grade_tier}
                        </td>
                        <td data-label="Momentum" className="py-2.5 px-4 text-slate-300">
                          {p.academic_momentum.toFixed(1)}
                        </td>
                        <td data-label="Engagement" className="py-2.5 px-4 text-emerald-300">
                          {(p.engagement_score * 100).toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
