import React, { useState } from 'react';
import { ModelMetadata, RuntimeTelemetry, AvailableModel } from '../types';
import { X, Sliders, Cpu, HardDrive, RefreshCw, AlertTriangle } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  model: ModelMetadata | null;
  runtime: RuntimeTelemetry | null;
  availableModels: AvailableModel[];
  onConfigure: (config: { model_path?: string; threads?: number; backend?: string }) => Promise<void>;
  onReload: () => Promise<void>;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  model,
  runtime,
  availableModels,
  onConfigure,
  onReload,
}) => {
  const [selectedModelPath, setSelectedModelPath] = useState(model?.path || '');
  const [selectedBackend, setSelectedBackend] = useState(runtime?.active_backend || 'NEON');
  const [threadCount, setThreadCount] = useState(runtime?.num_threads || 4);
  const [isApplying, setIsApplying] = useState(false);
  const [statusMsg, setStatusMsg] = useState<{ text: string; type: 'success' | 'error' } | null>(null);

  if (!isOpen) return null;

  const handleSave = async () => {
    setIsApplying(true);
    setStatusMsg(null);
    try {
      await onConfigure({
        model_path: selectedModelPath !== model?.path ? selectedModelPath : undefined,
        threads: threadCount !== runtime?.num_threads ? threadCount : undefined,
        backend: selectedBackend !== runtime?.active_backend ? selectedBackend : undefined,
      });
      setStatusMsg({ text: 'Engine configuration updated successfully.', type: 'success' });
      setTimeout(() => {
        onClose();
      }, 700);
    } catch (err: any) {
      setStatusMsg({ text: err.message || 'Failed to apply configuration', type: 'error' });
    } finally {
      setIsApplying(false);
    }
  };

  const handleFullReload = async () => {
    setIsApplying(true);
    setStatusMsg(null);
    try {
      await onReload();
      setStatusMsg({ text: 'C++ engine restarted successfully.', type: 'success' });
      setTimeout(() => {
        onClose();
      }, 700);
    } catch (err: any) {
      setStatusMsg({ text: err.message || 'Failed to reload engine', type: 'error' });
    } finally {
      setIsApplying(false);
    }
  };

  const maxHwThreads = runtime?.hardware_threads || 8;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="terminal-panel max-w-xl w-full bg-[#0d1117] border border-neutral-700 shadow-2xl rounded-lg overflow-hidden flex flex-col">
        {/* Modal Header */}
        <div className="p-4 border-b border-neutral-800 flex items-center justify-between bg-neutral-900/60">
          <div className="flex items-center gap-2 text-neutral-100 font-mono text-sm font-bold">
            <Sliders className="w-4 h-4 text-emerald-400" />
            Runtime Configuration & Engine Controls
          </div>
          <button
            onClick={onClose}
            className="text-neutral-400 hover:text-neutral-200 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-6 text-xs font-mono text-neutral-300">
          {statusMsg && (
            <div
              className={`p-3 rounded border text-xs ${
                statusMsg.type === 'success'
                  ? 'bg-emerald-950/40 text-emerald-400 border-emerald-800/60'
                  : 'bg-red-950/40 text-red-400 border-red-800/60'
              }`}
            >
              {statusMsg.text}
            </div>
          )}

          {/* Model Selection */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-neutral-200 font-bold">
              <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
              Active .ie Model Checkpoint
            </label>
            <select
              value={selectedModelPath}
              onChange={(e) => setSelectedModelPath(e.target.value)}
              className="w-full bg-neutral-900 border border-neutral-800 rounded p-2 text-xs text-neutral-200 focus:outline-none focus:border-emerald-500 font-mono"
            >
              {availableModels.map((m) => (
                <option key={m.path} value={m.path}>
                  {m.path} ({m.format}, {(m.size_bytes / (1024 * 1024)).toFixed(1)} MB)
                </option>
              ))}
              {!availableModels.some((m) => m.path === selectedModelPath) && (
                <option value={selectedModelPath}>{selectedModelPath}</option>
              )}
            </select>
            <p className="text-[11px] text-neutral-500">
              Switching models triggers a fresh zero-copy mmap inside the persistent C++ process.
            </p>
          </div>

          {/* SIMD Backend */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-neutral-200 font-bold">
              <Cpu className="w-3.5 h-3.5 text-emerald-400" />
              SIMD Instruction Backend
            </label>
            <div className="grid grid-cols-3 gap-2">
              <button
                type="button"
                disabled={!runtime?.neon_available}
                onClick={() => setSelectedBackend('NEON')}
                className={`p-2 rounded border text-center transition-all ${
                  selectedBackend === 'NEON'
                    ? 'border-emerald-500 bg-emerald-950/40 text-emerald-300 font-bold'
                    : runtime?.neon_available
                    ? 'border-neutral-800 bg-neutral-900 text-neutral-400 hover:border-neutral-700'
                    : 'border-neutral-900 bg-neutral-950 text-neutral-600 cursor-not-allowed'
                }`}
              >
                ARM NEON
                <div className="text-[10px] font-normal text-neutral-500">
                  {runtime?.neon_available ? 'Available' : 'Unsupported'}
                </div>
              </button>

              <button
                type="button"
                disabled={!runtime?.avx2_available}
                onClick={() => setSelectedBackend('AVX2')}
                className={`p-2 rounded border text-center transition-all ${
                  selectedBackend === 'AVX2'
                    ? 'border-emerald-500 bg-emerald-950/40 text-emerald-300 font-bold'
                    : runtime?.avx2_available
                    ? 'border-neutral-800 bg-neutral-900 text-neutral-400 hover:border-neutral-700'
                    : 'border-neutral-900 bg-neutral-950 text-neutral-600 cursor-not-allowed'
                }`}
              >
                x86 AVX2 / FMA
                <div className="text-[10px] font-normal text-neutral-500">
                  {runtime?.avx2_available ? 'Available' : 'Unsupported'}
                </div>
              </button>

              <button
                type="button"
                onClick={() => setSelectedBackend('Scalar')}
                className={`p-2 rounded border text-center transition-all ${
                  selectedBackend === 'Scalar'
                    ? 'border-emerald-500 bg-emerald-950/40 text-emerald-300 font-bold'
                    : 'border-neutral-800 bg-neutral-900 text-neutral-400 hover:border-neutral-700'
                }`}
              >
                Scalar Reference
                <div className="text-[10px] font-normal text-neutral-500">Always Available</div>
              </button>
            </div>
          </div>

          {/* Thread Concurrency */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="text-neutral-200 font-bold">Inference Worker Threads</label>
              <span className="text-emerald-400 font-bold">{threadCount} Threads</span>
            </div>
            <input
              type="range"
              min="1"
              max={maxHwThreads}
              step="1"
              value={threadCount}
              onChange={(e) => setThreadCount(parseInt(e.target.value, 10))}
              className="w-full accent-emerald-500 bg-neutral-800 h-1.5 rounded cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-neutral-500">
              <span>1 Thread (Single)</span>
              <span>Hardware Concurrency: {maxHwThreads} cores</span>
            </div>
          </div>

          {/* Danger Zone: Engine Restart */}
          <div className="p-3 rounded border border-amber-900/40 bg-amber-950/20 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0" />
              <div>
                <div className="text-neutral-200 font-semibold">Restart C++ Subprocess</div>
                <div className="text-[10px] text-neutral-400">
                  Re-spawns <code>inference-engine --studio-worker</code> and purges memory.
                </div>
              </div>
            </div>
            <button
              onClick={handleFullReload}
              disabled={isApplying}
              className="px-3 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-xs border border-neutral-700 flex items-center gap-1.5 transition-colors disabled:opacity-50"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Restart
            </button>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-neutral-800 bg-neutral-900/60 flex items-center justify-end gap-3 font-mono text-xs">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={isApplying}
            className="px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500 text-black font-bold transition-colors disabled:opacity-50 flex items-center gap-1.5"
          >
            {isApplying ? 'Applying...' : 'Apply Changes'}
          </button>
        </div>
      </div>
    </div>
  );
};

