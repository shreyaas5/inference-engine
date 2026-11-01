import React, { useMemo } from 'react';
import {
  Activity,
  Gauge,
  Zap,
  Timer,
  Cpu,
  Layers,
  Database,
  Info,
  TrendingUp,
} from 'lucide-react';
import { GenerationMetrics, RuntimeTelemetry, ModelMetadata } from '../types/index.js';

interface ObservabilityPanelProps {
  metrics: GenerationMetrics | null;
  runtime: RuntimeTelemetry | null;
  model: ModelMetadata | null;
  instantTpsHistory: number[];
  isGenerating: boolean;
}

export const ObservabilityPanel: React.FC<ObservabilityPanelProps> = ({
  metrics,
  runtime,
  model,
  instantTpsHistory,
  isGenerating,
}) => {
  // SVG Throughput Chart computation
  const chartPoints = useMemo(() => {
    if (!instantTpsHistory || instantTpsHistory.length < 2) return '';
    const maxVal = Math.max(45, ...instantTpsHistory);
    const width = 280;
    const height = 80;
    const step = width / (instantTpsHistory.length - 1);

    return instantTpsHistory
      .map((val, idx) => {
        const x = idx * step;
        const y = height - (Math.min(val, maxVal) / maxVal) * (height - 10) - 5;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }, [instantTpsHistory]);

  const currentDecodeTps = metrics?.decode_tps || 0;
  const currentInstantTps = metrics?.instant_tps || currentDecodeTps;
  const ttft = metrics?.ttft_ms;
  const promptTps = metrics?.prompt_tps;
  const currentRss = runtime?.current_rss_mib || metrics?.current_rss_mib || 0;
  const peakRss = runtime?.peak_rss_mib || metrics?.peak_rss_mib || 0;
  const cpuPct = metrics?.cpu_utilization_pct || 0;
  const threads = runtime?.num_threads || metrics?.threads || 4;
  const backend = runtime?.active_backend || metrics?.backend || 'NEON';
  const genTokens = metrics?.generated_tokens || 0;
  const totalMs = metrics?.total_ms || 0;
  const reusedKv = metrics?.reused_kv_tokens || 0;
  const promptTokens = metrics?.prompt_tokens || 0;

  return (
    <aside className="w-80 bg-[#0a0d12] border-l border-[#1a222f] flex flex-col justify-between overflow-y-auto select-none font-mono">
      {/* Header */}
      <div>
        <div className="p-3 border-b border-[#1a222f] flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-bold text-slate-200">
            <Activity className="w-4 h-4 text-emerald-400" />
            <span>LIVE OBSERVABILITY</span>
          </div>
          <span className="text-[10px] text-slate-400 bg-[#141a24] px-1.5 py-0.5 rounded border border-[#212b3a]">
            {isGenerating ? (
              <span className="text-emerald-400 flex items-center gap-1 font-bold">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                ACTIVE
              </span>
            ) : (
              'READY'
            )}
          </span>
        </div>

        {/* Primary Gauges (2x2 Grid) */}
        <div className="p-3 grid grid-cols-2 gap-2 border-b border-[#1a222f]">
          {/* Gauge 1: Decode Throughput */}
          <div className="bg-[#0f141c] border border-[#1b2432] p-2.5 rounded relative group">
            <div className="flex items-center justify-between text-[10px] text-slate-400">
              <span className="flex items-center gap-1">
                <Gauge className="w-3 h-3 text-emerald-400" />
                <span>DECODE TPS</span>
              </span>
              <span title="Autoregressive generation throughput: (generated_tokens - 1) / decode_seconds">
                <Info className="w-2.5 h-2.5 text-slate-600 group-hover:text-slate-400" />
              </span>
            </div>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-xl font-bold text-emerald-400 font-mono">
                {currentDecodeTps > 0 ? currentDecodeTps.toFixed(1) : '—'}
              </span>
              <span className="text-[10px] text-slate-500">tok/s</span>
            </div>
            <div className="text-[9px] text-slate-500 mt-0.5 truncate">
              {currentInstantTps > 0 ? `Instant: ${currentInstantTps.toFixed(1)} tok/s` : 'Steady State'}
            </div>
          </div>

          {/* Gauge 2: Time To First Token */}
          <div className="bg-[#0f141c] border border-[#1b2432] p-2.5 rounded relative group">
            <div className="flex items-center justify-between text-[10px] text-slate-400">
              <span className="flex items-center gap-1">
                <Timer className="w-3 h-3 text-cyan-400" />
                <span>TTFT</span>
              </span>
              <span title="Time from request start to first sampled token, including prompt prefill">
                <Info className="w-2.5 h-2.5 text-slate-600 group-hover:text-slate-400" />
              </span>
            </div>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-xl font-bold text-cyan-400 font-mono">
                {ttft ? ttft.toFixed(0) : '—'}
              </span>
              <span className="text-[10px] text-slate-500">ms</span>
            </div>
            <div className="text-[9px] text-slate-500 mt-0.5 truncate">
              Prefill: {metrics?.prefill_ms ? `${metrics.prefill_ms.toFixed(0)} ms` : '—'}
            </div>
          </div>

          {/* Gauge 3: Prompt Throughput */}
          <div className="bg-[#0f141c] border border-[#1b2432] p-2.5 rounded relative group">
            <div className="flex items-center justify-between text-[10px] text-slate-400">
              <span className="flex items-center gap-1">
                <Zap className="w-3 h-3 text-amber-400" />
                <span>PREFILL TPS</span>
              </span>
              <span title="Prompt processing throughput: prompt_tokens / prefill_seconds">
                <Info className="w-2.5 h-2.5 text-slate-600 group-hover:text-slate-400" />
              </span>
            </div>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-xl font-bold text-amber-400 font-mono">
                {promptTps && promptTps > 0 ? promptTps.toFixed(1) : '—'}
              </span>
              <span className="text-[10px] text-slate-500">tok/s</span>
            </div>
            <div className="text-[9px] text-slate-500 mt-0.5 truncate">
              {reusedKv > 0 ? `Reused: ${reusedKv} tok` : 'Full Prefill'}
            </div>
          </div>

          {/* Gauge 4: Peak Resident Memory */}
          <div className="bg-[#0f141c] border border-[#1b2432] p-2.5 rounded relative group">
            <div className="flex items-center justify-between text-[10px] text-slate-400">
              <span className="flex items-center gap-1">
                <Database className="w-3 h-3 text-purple-400" />
                <span>PEAK RSS</span>
              </span>
              <span title="Peak resident physical RAM measured via mach_task_basic_info / /proc/self/statm">
                <Info className="w-2.5 h-2.5 text-slate-600 group-hover:text-slate-400" />
              </span>
            </div>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-xl font-bold text-purple-400 font-mono">
                {peakRss > 0 ? peakRss.toFixed(1) : '—'}
              </span>
              <span className="text-[10px] text-slate-500">MiB</span>
            </div>
            <div className="text-[9px] text-slate-500 mt-0.5 truncate">
              Live RSS: {currentRss > 0 ? `${currentRss.toFixed(1)} MiB` : '—'}
            </div>
          </div>
        </div>

        {/* Live Throughput Chart */}
        <div className="p-3 border-b border-[#1a222f] space-y-2">
          <div className="flex items-center justify-between text-[10px] text-slate-400">
            <span className="flex items-center gap-1.5">
              <TrendingUp className="w-3 h-3 text-emerald-400" />
              <span>LIVE DECODE THROUGHPUT OVER TIME</span>
            </span>
            <span className="text-[9px] text-slate-500">
              {instantTpsHistory.length} samples
            </span>
          </div>

          <div className="bg-[#0b0f15] border border-[#1b2330] p-2 rounded relative h-24 flex items-end">
            {chartPoints ? (
              <svg className="w-full h-full overflow-visible" viewBox="0 0 280 80">
                <defs>
                  <linearGradient id="tpsGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#10b981" stopOpacity="0.4" />
                    <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
                  </linearGradient>
                </defs>
                {/* Horizontal reference grid lines */}
                <line x1="0" y1="20" x2="280" y2="20" stroke="#1f2937" strokeDasharray="2,2" />
                <line x1="0" y1="50" x2="280" y2="50" stroke="#1f2937" strokeDasharray="2,2" />
                {/* Area fill */}
                <polygon
                  points={`0,80 ${chartPoints} 280,80`}
                  fill="url(#tpsGradient)"
                />
                {/* Line stroke */}
                <polyline
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="1.8"
                  points={chartPoints}
                />
              </svg>
            ) : (
              <div className="w-full h-full flex items-center justify-center text-[10px] text-slate-600">
                Awaiting token generation stream...
              </div>
            )}
            <div className="absolute top-1 left-2 text-[9px] text-slate-600 font-mono">
              ~35 tok/s target
            </div>
          </div>
        </div>

        {/* Technical Telemetry Table */}
        <div className="p-3 space-y-1.5 text-xs">
          <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">
            DETAILED RUNTIME TELEMETRY
          </div>

          <div className="space-y-1 text-slate-400 text-[11px]">
            <div className="flex justify-between py-0.5 border-b border-[#141b25]">
              <span className="text-slate-500">SIMD Vector Backend:</span>
              <span className="text-emerald-400 font-bold">{backend}</span>
            </div>
            <div className="flex justify-between py-0.5 border-b border-[#141b25]">
              <span className="text-slate-500">Worker Threads:</span>
              <span className="text-slate-200">{threads} physical</span>
            </div>
            <div className="flex justify-between py-0.5 border-b border-[#141b25]">
              <span className="text-slate-500">CPU Utilisation:</span>
              <span className="text-slate-200">
                {cpuPct > 0 ? `${cpuPct.toFixed(1)}%` : '—'}
              </span>
            </div>
            <div className="flex justify-between py-0.5 border-b border-[#141b25]">
              <span className="text-slate-500">Tokens Generated:</span>
              <span className="text-cyan-400 font-bold">{genTokens}</span>
            </div>
            <div className="flex justify-between py-0.5 border-b border-[#141b25]">
              <span className="text-slate-500">Prompt Tokens (Turn):</span>
              <span className="text-slate-200">{promptTokens}</span>
            </div>
            <div className="flex justify-between py-0.5 border-b border-[#141b25]">
              <span className="text-slate-500">KV Cache Reused:</span>
              <span className="text-emerald-400">{reusedKv} tokens</span>
            </div>
            <div className="flex justify-between py-0.5 border-b border-[#141b25]">
              <span className="text-slate-500">Total Generation Time:</span>
              <span className="text-slate-200">
                {totalMs > 0 ? `${(totalMs / 1000).toFixed(2)} s` : '—'}
              </span>
            </div>
            <div className="flex justify-between py-0.5 border-b border-[#141b25]">
              <span className="text-slate-500">KV Memory Allocated:</span>
              <span className="text-slate-200">
                {runtime?.kv_cache_size_mib ? `${runtime.kv_cache_size_mib.toFixed(1)} MiB` : '88.0 MiB'}
              </span>
            </div>
            <div className="flex justify-between py-0.5 border-b border-[#141b25]">
              <span className="text-slate-500">Host Architecture:</span>
              <span className="text-slate-300 truncate max-w-[150px]" title={runtime?.os_arch}>
                {runtime?.os_arch || 'ARM64'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Footer / Metric Definitions Note */}
      <div className="p-3 border-t border-[#1a222f] bg-[#080b0f] text-[10px] text-slate-500 leading-tight">
        Telemetry streamed directly from C++17 <code className="text-emerald-400">inference_engine::studio::StudioWorker</code> via sub-millisecond process pipes.
      </div>
    </aside>
  );
};
