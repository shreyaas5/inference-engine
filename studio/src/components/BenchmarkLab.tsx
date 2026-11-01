import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Play,
  CheckCircle2,
  AlertCircle,
  TrendingUp,
  Database,
  Timer,
  Layers,
  ArrowRight,
  History,
  RotateCcw,
} from 'lucide-react';
import {
  BenchmarkRecord,
  ModelMetadata,
  RuntimeTelemetry,
  AvailableModel,
} from '../types/index.js';
import { startBenchmark, fetchBenchmarkHistory } from '../api/client.js';

interface BenchmarkLabProps {
  model: ModelMetadata | null;
  runtime: RuntimeTelemetry | null;
  availableModels: AvailableModel[];
  isBenchmarking: boolean;
  benchmarkProgress: { completed: number; total: number; latest?: any } | null;
  onBenchmarkDone: (record: BenchmarkRecord) => void;
}

export const BenchmarkLab: React.FC<BenchmarkLabProps> = ({
  model,
  runtime,
  availableModels,
  isBenchmarking,
  benchmarkProgress,
  onBenchmarkDone,
}) => {
  const [selectedModelPath, setSelectedModelPath] = useState(
    model?.path || 'models/tinyllama-1.1b-q4_0.ie'
  );
  const [threads, setThreads] = useState(runtime?.num_threads || 4);
  const [runs, setRuns] = useState(3);
  const [maxTokens, setMaxTokens] = useState(32);
  const [prompt, setPrompt] = useState('Explain entropy.');
  const [history, setHistory] = useState<BenchmarkRecord[]>([]);
  const [selectedRecord, setSelectedRecord] = useState<BenchmarkRecord | null>(null);
  const [compareRecord, setCompareRecord] = useState<BenchmarkRecord | null>(null);
  const [statusMessage, setStatusMessage] = useState('');

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const data = await fetchBenchmarkHistory();
      if (data && data.history) {
        setHistory(data.history);
        if (data.history.length > 0 && !selectedRecord) {
          setSelectedRecord(data.history[0]);
          if (data.history.length > 1) {
            setCompareRecord(data.history[1]);
          }
        }
      }
    } catch {
      // ignore
    }
  };

  const handleRunBenchmark = async () => {
    setStatusMessage('Initiating benchmark execution...');
    try {
      await startBenchmark({
        model_path: selectedModelPath,
        threads,
        runs,
        max_tokens: maxTokens,
        prompt,
      });
      setStatusMessage('');
      await loadHistory();
    } catch (err: any) {
      setStatusMessage(`Error: ${err.message}`);
    }
  };

  const activeRec = selectedRecord || history[0] || null;

  return (
    <div className="flex-1 flex flex-col h-full bg-[#080b0f] font-mono overflow-y-auto p-4 space-y-4">
      {/* Header */}
      <div className="p-3 bg-[#0d1219] border border-[#1a2333] rounded-lg flex items-center justify-between">
        <div>
          <h2 className="text-sm font-bold text-slate-100 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <span>BENCHMARK LAB &amp; HARDWARE PROFILER</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Measure statistical throughput, TTFT latency, and memory utilization across multi-iteration runs.
          </p>
        </div>

        <button
          onClick={loadHistory}
          className="flex items-center gap-1.5 px-2.5 py-1 text-xs bg-[#131b26] hover:bg-[#1a2434] text-slate-300 border border-[#223044] rounded transition cursor-pointer"
        >
          <RotateCcw className="w-3 h-3" />
          <span>Refresh Records</span>
        </button>
      </div>

      {/* Benchmark Configurator Card */}
      <div className="p-4 bg-[#0d1219] border border-[#1a2333] rounded-lg space-y-4">
        <div className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
          <span>1. Execution Configuration</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
          {/* Model Selection */}
          <div>
            <label className="block text-slate-400 text-[10px] mb-1">Target Model File</label>
            <select
              value={selectedModelPath}
              onChange={(e) => setSelectedModelPath(e.target.value)}
              disabled={isBenchmarking}
              className="w-full bg-[#111722] border border-[#202c3e] text-slate-200 text-xs p-2 rounded focus:outline-none focus:border-cyan-500 font-mono"
            >
              {availableModels.length > 0 ? (
                availableModels.map((m) => (
                  <option key={m.path} value={m.path}>
                    {m.format} ({(m.size_bytes / (1024 * 1024)).toFixed(0)} MB) - {m.path}
                  </option>
                ))
              ) : (
                <option value="models/tinyllama-1.1b-q4_0.ie">
                  Q4_0 (619 MB) - models/tinyllama-1.1b-q4_0.ie
                </option>
              )}
            </select>
          </div>

          {/* Worker Threads */}
          <div>
            <label className="block text-slate-400 text-[10px] mb-1">
              Worker Threads ({threads} Threads)
            </label>
            <input
              type="range"
              min="1"
              max={runtime?.hardware_threads || 8}
              step="1"
              value={threads}
              onChange={(e) => setThreads(parseInt(e.target.value, 10))}
              disabled={isBenchmarking}
              className="w-full accent-cyan-500 cursor-pointer h-2 bg-[#1a2332] rounded mt-2"
            />
          </div>

          {/* Iteration Count */}
          <div>
            <label className="block text-slate-400 text-[10px] mb-1">
              Benchmark Runs (N = {runs})
            </label>
            <input
              type="range"
              min="1"
              max="10"
              step="1"
              value={runs}
              onChange={(e) => setRuns(parseInt(e.target.value, 10))}
              disabled={isBenchmarking}
              className="w-full accent-cyan-500 cursor-pointer h-2 bg-[#1a2332] rounded mt-2"
            />
          </div>

          {/* Max Tokens */}
          <div>
            <label className="block text-slate-400 text-[10px] mb-1">
              Tokens / Run ({maxTokens} tok)
            </label>
            <input
              type="range"
              min="16"
              max="128"
              step="16"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value, 10))}
              disabled={isBenchmarking}
              className="w-full accent-cyan-500 cursor-pointer h-2 bg-[#1a2332] rounded mt-2"
            />
          </div>
        </div>

        {/* Prompt Input */}
        <div>
          <label className="block text-slate-400 text-[10px] mb-1">Benchmark Standard Prompt</label>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={isBenchmarking}
            placeholder="Enter benchmark prompt..."
            className="w-full bg-[#111722] border border-[#202c3e] text-slate-200 text-xs p-2 rounded focus:outline-none focus:border-cyan-500 font-mono"
          />
        </div>

        {/* Run Button & Live Progress */}
        <div className="flex items-center justify-between pt-2">
          <div className="text-xs text-slate-400">
            {isBenchmarking ? (
              <span className="flex items-center gap-2 text-cyan-400">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
                <span>
                  Executing Run {benchmarkProgress?.completed ?? 1} of{' '}
                  {benchmarkProgress?.total ?? runs}...
                </span>
              </span>
            ) : statusMessage ? (
              <span className="text-amber-400">{statusMessage}</span>
            ) : (
              <span className="text-slate-500">
                Ready to execute {runs} iterations with thread pool isolation.
              </span>
            )}
          </div>

          <button
            onClick={handleRunBenchmark}
            disabled={isBenchmarking}
            className="flex items-center gap-2 px-4 py-2 bg-cyan-500 hover:bg-cyan-600 disabled:bg-[#1a2332] text-slate-950 disabled:text-slate-600 rounded text-xs font-bold transition cursor-pointer disabled:cursor-not-allowed shadow-[0_0_12px_rgba(6,182,212,0.25)] active:scale-95"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>EXECUTE BENCHMARK SUITE</span>
          </button>
        </div>
      </div>

      {/* Selected Benchmark Results Card */}
      {activeRec && (
        <div className="p-4 bg-[#0d1219] border border-[#1a2333] rounded-lg space-y-4">
          <div className="flex items-center justify-between border-b border-[#1b2536] pb-2">
            <div>
              <div className="text-xs font-bold text-slate-200 flex items-center gap-2">
                <span>{activeRec.label}</span>
                {activeRec.is_historical ? (
                  <span className="text-[10px] bg-amber-950/40 text-amber-300 border border-amber-800/60 px-1.5 py-0.2 rounded font-mono">
                    HISTORICAL VERIFICATION
                  </span>
                ) : (
                  <span className="text-[10px] bg-emerald-950/40 text-emerald-300 border border-emerald-800/60 px-1.5 py-0.2 rounded font-mono">
                    LIVE MEASUREMENT
                  </span>
                )}
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                Timestamp: {activeRec.timestamp} | Model: {activeRec.model_path} | Backend:{' '}
                {activeRec.backend} ({activeRec.threads} Threads)
              </div>
            </div>

            <div className="text-right text-[10px] text-slate-400 font-mono">
              <div>Host: {activeRec.os_arch}</div>
              <div className="text-slate-500 truncate max-w-[200px]" title={activeRec.compiler}>
                {activeRec.compiler}
              </div>
            </div>
          </div>

          {/* 4 Stat Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="bg-[#101622] p-3 rounded border border-[#1d293d]">
              <div className="text-[10px] text-slate-400">DECODE THROUGHPUT</div>
              <div className="text-lg font-bold text-emerald-400 mt-0.5">
                {activeRec.decode_tps.mean.toFixed(2)}{' '}
                <span className="text-[10px] text-slate-500">tok/s</span>
              </div>
              <div className="text-[10px] text-slate-500 mt-1">
                Median: {activeRec.decode_tps.median.toFixed(2)} | Min:{' '}
                {activeRec.decode_tps.min.toFixed(2)} | Max: {activeRec.decode_tps.max.toFixed(2)}
              </div>
            </div>

            <div className="bg-[#101622] p-3 rounded border border-[#1d293d]">
              <div className="text-[10px] text-slate-400">PROMPT THROUGHPUT</div>
              <div className="text-lg font-bold text-cyan-400 mt-0.5">
                {activeRec.prompt_tps.mean.toFixed(2)}{' '}
                <span className="text-[10px] text-slate-500">tok/s</span>
              </div>
              <div className="text-[10px] text-slate-500 mt-1">
                Median: {activeRec.prompt_tps.median.toFixed(2)} | Min:{' '}
                {activeRec.prompt_tps.min.toFixed(2)} | Max: {activeRec.prompt_tps.max.toFixed(2)}
              </div>
            </div>

            <div className="bg-[#101622] p-3 rounded border border-[#1d293d]">
              <div className="text-[10px] text-slate-400">TIME TO FIRST TOKEN (TTFT)</div>
              <div className="text-lg font-bold text-amber-400 mt-0.5">
                {activeRec.ttft_ms.mean.toFixed(1)}{' '}
                <span className="text-[10px] text-slate-500">ms</span>
              </div>
              <div className="text-[10px] text-slate-500 mt-1">
                Median: {activeRec.ttft_ms.median.toFixed(1)} | Min:{' '}
                {activeRec.ttft_ms.min.toFixed(1)} | Max: {activeRec.ttft_ms.max.toFixed(1)}
              </div>
            </div>

            <div className="bg-[#101622] p-3 rounded border border-[#1d293d]">
              <div className="text-[10px] text-slate-400">PEAK RESIDENT RAM (RSS)</div>
              <div className="text-lg font-bold text-purple-400 mt-0.5">
                {activeRec.peak_rss_mib.toFixed(1)}{' '}
                <span className="text-[10px] text-slate-500">MiB</span>
              </div>
              <div className="text-[10px] text-slate-500 mt-1">
                Model Load: {activeRec.model_load_ms.toFixed(2)} ms (zero-copy mmap)
              </div>
            </div>
          </div>

          {/* Individual Runs Breakdown (if available) */}
          {activeRec.individual_runs && activeRec.individual_runs.length > 0 && (
            <div className="space-y-1.5 pt-2">
              <div className="text-[10px] font-bold text-slate-400 uppercase">
                Per-Iteration Measurements
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[11px] text-slate-300 border-collapse">
                  <thead>
                    <tr className="border-b border-[#1f2a3a] text-slate-500 text-[10px]">
                      <th className="py-1 px-2">RUN</th>
                      <th className="py-1 px-2">TTFT</th>
                      <th className="py-1 px-2">PREFILL TPS</th>
                      <th className="py-1 px-2">DECODE TPS</th>
                      <th className="py-1 px-2">TOTAL TIME</th>
                      <th className="py-1 px-2">TOKENS</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeRec.individual_runs.map((r) => (
                      <tr key={r.run_index} className="border-b border-[#141b25] hover:bg-[#121824]">
                        <td className="py-1 px-2 font-bold text-slate-400">#{r.run_index}</td>
                        <td className="py-1 px-2 text-amber-400">{r.ttft_ms.toFixed(1)} ms</td>
                        <td className="py-1 px-2 text-cyan-400">{r.prompt_tps.toFixed(1)} tok/s</td>
                        <td className="py-1 px-2 text-emerald-400 font-bold">
                          {r.decode_tps.toFixed(2)} tok/s
                        </td>
                        <td className="py-1 px-2 text-slate-400">{(r.total_ms / 1000).toFixed(2)} s</td>
                        <td className="py-1 px-2 text-slate-300">{r.generated_tokens}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Benchmark History & Comparison Table */}
      <div className="p-4 bg-[#0d1219] border border-[#1a2333] rounded-lg space-y-3">
        <div className="flex items-center justify-between">
          <div className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
            <History className="w-3.5 h-3.5 text-cyan-400" />
            <span>2. Historical Records &amp; Side-by-Side Comparison</span>
          </div>
          <span className="text-[10px] text-slate-500">
            Click row to select active record
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300 border-collapse">
            <thead>
              <tr className="border-b border-[#1f2a3a] text-slate-500 text-[10px]">
                <th className="py-1.5 px-2">RECORD LABEL</th>
                <th className="py-1.5 px-2">FORMAT</th>
                <th className="py-1.5 px-2">BACKEND</th>
                <th className="py-1.5 px-2">THREADS</th>
                <th className="py-1.5 px-2">DECODE (MEAN)</th>
                <th className="py-1.5 px-2">TTFT (MEAN)</th>
                <th className="py-1.5 px-2">PEAK RSS</th>
                <th className="py-1.5 px-2">TYPE</th>
              </tr>
            </thead>
            <tbody>
              {history.map((rec) => {
                const isSelected = selectedRecord?.id === rec.id;
                return (
                  <tr
                    key={rec.id}
                    onClick={() => setSelectedRecord(rec)}
                    className={`border-b border-[#141b25] transition cursor-pointer text-[11px] ${
                      isSelected ? 'bg-[#151f2e] text-slate-100 font-semibold' : 'hover:bg-[#101622]'
                    }`}
                  >
                    <td className="py-1.5 px-2 text-slate-200">{rec.label}</td>
                    <td className="py-1.5 px-2 text-cyan-400 font-bold">{rec.format}</td>
                    <td className="py-1.5 px-2 text-emerald-400">{rec.backend}</td>
                    <td className="py-1.5 px-2">{rec.threads}T</td>
                    <td className="py-1.5 px-2 text-emerald-400 font-bold">
                      {rec.decode_tps.mean.toFixed(2)} tok/s
                    </td>
                    <td className="py-1.5 px-2 text-amber-400">{rec.ttft_ms.mean.toFixed(1)} ms</td>
                    <td className="py-1.5 px-2 text-purple-400">{rec.peak_rss_mib.toFixed(1)} MiB</td>
                    <td className="py-1.5 px-2">
                      {rec.is_historical ? (
                        <span className="text-[9px] px-1 py-0.2 rounded bg-amber-950/50 text-amber-400 border border-amber-800/50">
                          Historical
                        </span>
                      ) : (
                        <span className="text-[9px] px-1 py-0.2 rounded bg-emerald-950/50 text-emerald-400 border border-emerald-800/50">
                          Live
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
