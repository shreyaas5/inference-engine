import React from 'react';
import {
  Terminal,
  Activity,
  Cpu,
  Layers,
  FileText,
  Settings,
  Plus,
  Trash2,
  HardDrive,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Database,
} from 'lucide-react';
import { ModelMetadata, RuntimeTelemetry } from '../types/index.js';

interface NavigationSidebarProps {
  currentTab: 'chat' | 'benchmark' | 'inspector' | 'logs';
  onSelectTab: (tab: 'chat' | 'benchmark' | 'inspector' | 'logs') => void;
  onNewChat: () => void;
  onOpenSettings: () => void;
  engineConnected: boolean;
  engineState: string;
  model: ModelMetadata | null;
  runtime: RuntimeTelemetry | null;
  chatTurnCount: number;
  onClearContext: () => void;
}

export const NavigationSidebar: React.FC<NavigationSidebarProps> = ({
  currentTab,
  onSelectTab,
  onNewChat,
  onOpenSettings,
  engineConnected,
  engineState,
  model,
  runtime,
  chatTurnCount,
  onClearContext,
}) => {
  const getStatusBadge = () => {
    if (!engineConnected) {
      return (
        <span className="flex items-center gap-1.5 text-xs text-red-400 bg-red-950/40 border border-red-800/60 px-2 py-0.5 rounded">
          <XCircle className="w-3 h-3 text-red-400 animate-pulse" />
          OFFLINE
        </span>
      );
    }
    if (engineState === 'generating') {
      return (
        <span className="flex items-center gap-1.5 text-xs text-emerald-400 bg-emerald-950/40 border border-emerald-600/60 px-2 py-0.5 rounded shadow-[0_0_8px_rgba(16,185,129,0.3)]">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
          DECODING
        </span>
      );
    }
    if (engineState === 'benchmarking') {
      return (
        <span className="flex items-center gap-1.5 text-xs text-amber-400 bg-amber-950/40 border border-amber-600/60 px-2 py-0.5 rounded">
          <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
          BENCHMARKING
        </span>
      );
    }
    return (
      <span className="flex items-center gap-1.5 text-xs text-emerald-400 bg-emerald-950/30 border border-emerald-800/60 px-2 py-0.5 rounded">
        <CheckCircle2 className="w-3 h-3 text-emerald-400" />
        ONLINE
      </span>
    );
  };

  return (
    <aside className="w-64 bg-[#0a0d12] border-r border-[#1a222f] flex flex-col justify-between select-none">
      {/* Header / Brand */}
      <div>
        <div className="p-3 border-b border-[#1a222f] flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded bg-[#101722] border border-[#10b981]/40 flex items-center justify-center text-[#10b981] shadow-[0_0_12px_rgba(16,185,129,0.15)]">
              <Terminal className="w-4 h-4" />
            </div>
            <div>
              <div className="text-xs font-bold tracking-wider text-slate-100 flex items-center gap-1.5">
                <span>BIOS V2</span>
                <span className="text-[9px] px-1 py-0.2 bg-[#10b981]/20 text-[#10b981] border border-[#10b981]/40 rounded font-mono">
                  PS08
                </span>
              </div>
              <div className="text-[10px] text-slate-400 tracking-tight font-mono">
                INFERENCE STUDIO
              </div>
            </div>
          </div>
          {getStatusBadge()}
        </div>

        {/* Action: New Conversation */}
        <div className="p-3 space-y-2">
          <button
            onClick={onNewChat}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-[#10b981]/15 hover:bg-[#10b981]/25 border border-[#10b981]/40 hover:border-[#10b981]/70 text-[#10b981] text-xs font-mono font-medium rounded transition-all cursor-pointer shadow-[0_0_10px_rgba(16,185,129,0.1)] active:scale-[0.99]"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>NEW INFERENCE SESSION</span>
          </button>
        </div>

        {/* Navigation Tabs */}
        <nav className="px-2 py-1 space-y-1">
          <button
            onClick={() => onSelectTab('chat')}
            className={`w-full flex items-center gap-2.5 px-3 py-2 text-xs font-mono rounded transition-colors text-left cursor-pointer ${
              currentTab === 'chat'
                ? 'bg-[#151c27] text-emerald-400 border-l-2 border-emerald-400 font-semibold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-[#111620]'
            }`}
          >
            <Activity className="w-4 h-4 text-emerald-500" />
            <div className="flex-1 flex items-center justify-between">
              <span>Chat Playground</span>
              {chatTurnCount > 0 && (
                <span className="text-[10px] bg-[#1a2332] text-slate-400 px-1.5 py-0.2 rounded border border-[#232f42]">
                  {chatTurnCount} turns
                </span>
              )}
            </div>
          </button>

          <button
            onClick={() => onSelectTab('benchmark')}
            className={`w-full flex items-center gap-2.5 px-3 py-2 text-xs font-mono rounded transition-colors text-left cursor-pointer ${
              currentTab === 'benchmark'
                ? 'bg-[#151c27] text-cyan-400 border-l-2 border-cyan-400 font-semibold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-[#111620]'
            }`}
          >
            <Cpu className="w-4 h-4 text-cyan-500" />
            <span>Benchmark Lab</span>
          </button>

          <button
            onClick={() => onSelectTab('inspector')}
            className={`w-full flex items-center gap-2.5 px-3 py-2 text-xs font-mono rounded transition-colors text-left cursor-pointer ${
              currentTab === 'inspector'
                ? 'bg-[#151c27] text-purple-400 border-l-2 border-purple-400 font-semibold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-[#111620]'
            }`}
          >
            <Layers className="w-4 h-4 text-purple-400" />
            <span>Model Inspector</span>
          </button>

          <button
            onClick={() => onSelectTab('logs')}
            className={`w-full flex items-center gap-2.5 px-3 py-2 text-xs font-mono rounded transition-colors text-left cursor-pointer ${
              currentTab === 'logs'
                ? 'bg-[#151c27] text-amber-400 border-l-2 border-amber-400 font-semibold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-[#111620]'
            }`}
          >
            <FileText className="w-4 h-4 text-amber-500" />
            <span>Runtime Logs</span>
          </button>
        </nav>
      </div>

      {/* Model & System Telemetry Footer */}
      <div className="p-3 border-t border-[#1a222f] bg-[#080b0f] space-y-2.5 text-[11px] font-mono">
        <div className="flex items-center justify-between text-slate-400 border-b border-[#161d27] pb-1.5">
          <span className="flex items-center gap-1.5 text-slate-300">
            <HardDrive className="w-3 h-3 text-[#10b981]" />
            <span>LOADED ENGINE</span>
          </span>
          <span className="text-[10px] text-emerald-400 bg-emerald-950/60 px-1 rounded border border-emerald-800/40 font-bold">
            {model?.format || 'N/A'}
          </span>
        </div>

        <div className="space-y-1 text-slate-400">
          <div className="flex justify-between">
            <span className="text-slate-500">Model:</span>
            <span className="text-slate-300 truncate max-w-[140px]" title={model?.name || ''}>
              TinyLlama 1.1B
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Backend:</span>
            <span className="text-emerald-400 font-bold">{runtime?.active_backend || 'N/A'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Worker Threads:</span>
            <span className="text-slate-200">
              {runtime?.num_threads || 1} / {runtime?.hardware_threads || 1}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">KV Committed:</span>
            <span className="text-cyan-400">
              {runtime?.kv_seq_len ?? 0} / {runtime?.kv_max_seq_len ?? 2048} tok
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Peak RSS:</span>
            <span className="text-slate-200">
              {runtime?.peak_rss_mib ? `${runtime.peak_rss_mib.toFixed(1)} MiB` : 'N/A'}
            </span>
          </div>
        </div>

        <div className="pt-1 flex items-center gap-2">
          <button
            onClick={onClearContext}
            title="Purge C++ KV cache and conversation history"
            className="flex-1 flex items-center justify-center gap-1.5 py-1 px-2 bg-[#171e28] hover:bg-[#202937] text-slate-300 hover:text-red-400 border border-[#232e3d] rounded text-[10px] transition cursor-pointer"
          >
            <Trash2 className="w-3 h-3" />
            <span>RESET KV</span>
          </button>
          <button
            onClick={onOpenSettings}
            title="Configure model, threads, and backend"
            className="p-1 bg-[#171e28] hover:bg-[#202937] text-slate-300 hover:text-emerald-400 border border-[#232e3d] rounded text-[10px] transition cursor-pointer"
          >
            <Settings className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </aside>
  );
};

