import React, { useState, useRef, useEffect, useMemo } from 'react';
import { LogEntry } from '../types';
import { 
  Terminal, 
  Trash2, 
  Download, 
  Search, 
  ArrowDownCircle, 
  Filter, 
  CheckCircle, 
  AlertTriangle, 
  AlertCircle 
} from 'lucide-react';

interface RuntimeLogViewerProps {
  logs: LogEntry[];
  onClearLogs: () => Promise<void>;
}

export const RuntimeLogViewer: React.FC<RuntimeLogViewerProps> = ({ logs, onClearLogs }) => {
  const [levelFilter, setLevelFilter] = useState<'ALL' | 'INFO' | 'WARN' | 'ERROR'>('ALL');
  const [sourceFilter, setSourceFilter] = useState<'ALL' | 'cpp-engine' | 'studio-api'>('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [autoScroll, setAutoScroll] = useState(true);
  const logContainerRef = useRef<HTMLDivElement>(null);

  const filteredLogs = useMemo(() => {
    return logs.filter((log) => {
      const matchesLevel = levelFilter === 'ALL' || log.level === levelFilter;
      const matchesSource = sourceFilter === 'ALL' || log.source === sourceFilter;
      const matchesQuery =
        !searchQuery ||
        log.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
        log.source.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesLevel && matchesSource && matchesQuery;
    });
  }, [logs, levelFilter, sourceFilter, searchQuery]);

  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [filteredLogs, autoScroll]);

  const handleExportLogs = () => {
    const text = filteredLogs
      .map((l) => `[${l.timestamp}] [${l.level.padEnd(5)}] [${l.source}] ${l.message}`)
      .join('\n');
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `inference-engine-logs-${Date.now()}.log`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getLevelBadge = (level: string) => {
    switch (level) {
      case 'ERROR':
        return (
          <span className="flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded bg-red-950/60 text-red-400 border border-red-800/50">
            <AlertCircle className="w-3 h-3" />
            ERROR
          </span>
        );
      case 'WARN':
        return (
          <span className="flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-950/60 text-amber-400 border border-amber-800/50">
            <AlertTriangle className="w-3 h-3" />
            WARN
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-800/50">
            <CheckCircle className="w-3 h-3" />
            INFO
          </span>
        );
    }
  };

  return (
    <div className="h-full flex flex-col bg-[#080a0d] text-neutral-200">
      {/* Header & Controls Toolbar */}
      <div className="p-4 border-b border-neutral-800 flex flex-col md:flex-row items-center justify-between gap-3 bg-neutral-900/60">
        <div className="flex items-center gap-3">
          <Terminal className="w-5 h-5 text-emerald-400" />
          <div>
            <h2 className="text-sm font-bold text-neutral-100 font-mono flex items-center gap-2">
              Runtime Diagnostic Logs
              <span className="text-xs text-neutral-400 font-normal">
                ({filteredLogs.length} / {logs.length} entries)
              </span>
            </h2>
            <p className="text-[11px] text-neutral-500 font-mono">
              Live stdio & internal telemetry stream from custom C++17 engine
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 w-full md:w-auto">
          {/* Level Filter */}
          <div className="flex items-center gap-1.5 text-xs font-mono">
            <Filter className="w-3.5 h-3.5 text-neutral-500" />
            <select
              value={levelFilter}
              onChange={(e) => setLevelFilter(e.target.value as any)}
              className="bg-neutral-900 border border-neutral-800 text-neutral-200 rounded px-2 py-1 text-xs focus:outline-none focus:border-emerald-500 font-mono"
            >
              <option value="ALL">All Levels</option>
              <option value="INFO">INFO Only</option>
              <option value="WARN">WARN Only</option>
              <option value="ERROR">ERROR Only</option>
            </select>
          </div>

          {/* Source Filter */}
          <div className="flex items-center gap-1.5 text-xs font-mono">
            <select
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value as any)}
              className="bg-neutral-900 border border-neutral-800 text-neutral-200 rounded px-2 py-1 text-xs focus:outline-none focus:border-emerald-500 font-mono"
            >
              <option value="ALL">All Sources</option>
              <option value="cpp-engine">C++ Engine</option>
              <option value="studio-api">Studio Node API</option>
            </select>
          </div>

          {/* Search Input */}
          <div className="relative flex-1 md:w-52">
            <Search className="w-3.5 h-3.5 text-neutral-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search logs..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-neutral-900 border border-neutral-800 text-neutral-200 text-xs rounded pl-8 pr-3 py-1 focus:outline-none focus:border-emerald-500 font-mono"
            />
          </div>

          {/* Auto Scroll Toggle */}
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`flex items-center gap-1 px-2.5 py-1 rounded text-xs font-mono border transition-colors ${
              autoScroll
                ? 'bg-emerald-950/40 text-emerald-400 border-emerald-800/60'
                : 'bg-neutral-900 text-neutral-400 border-neutral-800'
            }`}
            title="Toggle automatic scrolling to bottom"
          >
            <ArrowDownCircle className="w-3.5 h-3.5" />
            Auto-Scroll
          </button>

          {/* Export Button */}
          <button
            onClick={handleExportLogs}
            className="flex items-center gap-1 px-2.5 py-1 rounded text-xs font-mono bg-neutral-900 hover:bg-neutral-800 text-neutral-300 border border-neutral-800 transition-colors"
            title="Download log file"
          >
            <Download className="w-3.5 h-3.5" />
            Export
          </button>

          {/* Clear Button */}
          <button
            onClick={onClearLogs}
            className="flex items-center gap-1 px-2.5 py-1 rounded text-xs font-mono bg-red-950/30 hover:bg-red-900/40 text-red-400 border border-red-800/40 transition-colors"
            title="Clear all displayed logs"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear
          </button>
        </div>
      </div>

      {/* Terminal Output Area */}
      <div
        ref={logContainerRef}
        className="flex-1 p-4 overflow-y-auto font-mono text-xs space-y-1 bg-[#05070a] select-text"
      >
        {filteredLogs.length === 0 ? (
          <div className="h-full flex items-center justify-center text-neutral-600 font-mono">
            No log entries recorded matching current filters.
          </div>
        ) : (
          filteredLogs.map((log) => {
            const timeStr = log.timestamp ? log.timestamp.split('T')[1].replace('Z', '') : '';
            return (
              <div
                key={log.id}
                className="flex items-start gap-2.5 py-1 px-2 rounded hover:bg-neutral-900/60 transition-colors font-mono leading-relaxed"
              >
                <span className="text-neutral-500 shrink-0 text-[11px] select-none">{timeStr}</span>
                <span className="shrink-0">{getLevelBadge(log.level)}</span>
                <span
                  className={`shrink-0 text-[10px] px-1.5 py-0.2 rounded border select-none ${
                    log.source === 'cpp-engine'
                      ? 'bg-blue-950/40 text-blue-400 border-blue-900/40'
                      : 'bg-neutral-800/80 text-neutral-300 border-neutral-700/60'
                  }`}
                >
                  {log.source}
                </span>
                <span
                  className={`flex-1 break-all ${
                    log.level === 'ERROR'
                      ? 'text-red-300 font-semibold'
                      : log.level === 'WARN'
                      ? 'text-amber-300'
                      : 'text-neutral-300'
                  }`}
                >
                  {log.message}
                </span>
              </div>
            );
          })
        )}
      </div>

      {/* Footer Status Bar */}
      <div className="px-4 py-2 border-t border-neutral-800/80 bg-neutral-900/50 flex items-center justify-between text-[11px] font-mono text-neutral-500">
        <div>
          Logging Channel: <span className="text-emerald-400">stdio:stdout/stderr</span> (JSON-Lines Protocol)
        </div>
        <div>
          Buffer size: {logs.length} / 500 lines
        </div>
      </div>
    </div>
  );
};
