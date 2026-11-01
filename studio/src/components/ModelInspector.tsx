import React, { useState, useMemo } from 'react';
import { ModelMetadata } from '../types';
import { 
  Database, 
  Layers, 
  Cpu, 
  Hash, 
  Search, 
  FileText, 
  HardDrive, 
  CheckCircle2, 
  Code2, 
  ChevronRight,
  Filter
} from 'lucide-react';

interface ModelInspectorProps {
  model: ModelMetadata | null;
}

export const ModelInspector: React.FC<ModelInspectorProps> = ({ model }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [dtypeFilter, setDtypeFilter] = useState<string>('ALL');

  const tensors = model?.tensors || [];

  const filteredTensors = useMemo(() => {
    return tensors.filter((t) => {
      const matchesSearch = t.name.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesDtype = dtypeFilter === 'ALL' || t.dtype === dtypeFilter;
      return matchesSearch && matchesDtype;
    });
  }, [tensors, searchQuery, dtypeFilter]);

  const uniqueDtypes = useMemo(() => {
    const set = new Set<string>();
    tensors.forEach((t) => set.add(t.dtype));
    return Array.from(set);
  }, [tensors]);

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat().format(num);
  };

  if (!model || !model.loaded) {
    return (
      <div className="h-full flex items-center justify-center p-8 bg-[#0a0d14]">
        <div className="terminal-panel max-w-md p-6 text-center border-amber-500/30">
          <Database className="w-12 h-12 text-amber-500 mx-auto mb-3 animate-pulse" />
          <h3 className="text-base font-semibold text-neutral-200">No Model Loaded</h3>
          <p className="text-xs text-neutral-400 mt-1">
            The C++ inference engine has not loaded an active model file.
          </p>
        </div>
      </div>
    );
  }

  const { config } = model;

  return (
    <div className="h-full flex flex-col overflow-y-auto bg-[#0a0d14] text-neutral-200 p-6 space-y-6">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-4 rounded-lg bg-neutral-900/60 border border-neutral-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="badge badge-green text-xs font-mono">.ie Format v1</span>
            <span className="badge badge-amber text-xs font-mono">{model.format}</span>
            <span className="text-xs text-neutral-400 font-mono">Magic: 0x464D4549 (IEMF)</span>
          </div>
          <h2 className="text-xl font-bold text-neutral-100 mt-2 font-mono">{model.name}</h2>
          <p className="text-xs text-neutral-400 mt-0.5 font-mono">{model.path}</p>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="text-right">
            <div className="text-neutral-400">Disk Footprint</div>
            <div className="text-emerald-400 font-semibold">{formatBytes(model.file_size_bytes)}</div>
          </div>
          <div className="h-8 w-px bg-neutral-800" />
          <div className="text-right">
            <div className="text-neutral-400">Total Weights</div>
            <div className="text-neutral-200 font-semibold">
              {(model.parameter_count / 1e9).toFixed(2)}B params
            </div>
          </div>
          <div className="h-8 w-px bg-neutral-800" />
          <div className="text-right">
            <div className="text-neutral-400">Loader Mode</div>
            <div className="text-cyan-400 font-semibold flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400" />
              {model.is_mmap ? 'Zero-Copy mmap' : 'Heap Buffer'}
            </div>
          </div>
        </div>
      </div>

      {/* Architecture & Hyperparameters Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="terminal-panel p-3.5">
          <div className="flex items-center justify-between text-neutral-400 mb-1">
            <span className="text-xs font-mono">Transformer Layers</span>
            <Layers className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="text-2xl font-bold font-mono text-neutral-100">{config.num_layers}</div>
          <div className="text-[10px] text-neutral-500 font-mono mt-0.5">Sequential Decoder Blocks</div>
        </div>

        <div className="terminal-panel p-3.5">
          <div className="flex items-center justify-between text-neutral-400 mb-1">
            <span className="text-xs font-mono">Hidden Dimension</span>
            <Hash className="w-4 h-4 text-cyan-500" />
          </div>
          <div className="text-2xl font-bold font-mono text-neutral-100">{config.hidden_dim}</div>
          <div className="text-[10px] text-neutral-500 font-mono mt-0.5">Intermediate: {config.intermediate_dim} (SwiGLU)</div>
        </div>

        <div className="terminal-panel p-3.5">
          <div className="flex items-center justify-between text-neutral-400 mb-1">
            <span className="text-xs font-mono">Attention Heads (GQA)</span>
            <Cpu className="w-4 h-4 text-amber-500" />
          </div>
          <div className="text-2xl font-bold font-mono text-neutral-100">
            {config.num_query_heads} <span className="text-sm font-normal text-neutral-400">Q</span> / {config.num_kv_heads} <span className="text-sm font-normal text-neutral-400">KV</span>
          </div>
          <div className="text-[10px] text-amber-400/80 font-mono mt-0.5">
            Group Size: {config.gqa_group_size}:1 (Head Dim: {config.head_dim})
          </div>
        </div>

        <div className="terminal-panel p-3.5">
          <div className="flex items-center justify-between text-neutral-400 mb-1">
            <span className="text-xs font-mono">Max Context / Vocab</span>
            <HardDrive className="w-4 h-4 text-purple-500" />
          </div>
          <div className="text-2xl font-bold font-mono text-neutral-100">{config.max_seq_len}</div>
          <div className="text-[10px] text-neutral-500 font-mono mt-0.5">
            Vocab: {formatNumber(config.vocab_size)} tokens
          </div>
        </div>
      </div>

      {/* Technical Specification Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* RoPE & Normalization */}
        <div className="terminal-panel p-4">
          <div className="flex items-center gap-2 text-xs font-mono text-neutral-300 font-semibold mb-3 border-b border-neutral-800 pb-2">
            <Code2 className="w-4 h-4 text-emerald-400" />
            Position & Normalization
          </div>
          <div className="space-y-2 text-xs font-mono">
            <div className="flex justify-between">
              <span className="text-neutral-400">RoPE Theta:</span>
              <span className="text-neutral-200">{config.rope_theta.toFixed(1)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">RMSNorm Epsilon:</span>
              <span className="text-neutral-200">{config.rms_norm_eps}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Position Indexing:</span>
              <span className="text-neutral-200">Continuous 0 .. N-1</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Feed-Forward Act:</span>
              <span className="text-emerald-400">SwiGLU (SiLU * Gate)</span>
            </div>
          </div>
        </div>

        {/* Quantisation & Storage Contract */}
        <div className="terminal-panel p-4">
          <div className="flex items-center gap-2 text-xs font-mono text-neutral-300 font-semibold mb-3 border-b border-neutral-800 pb-2">
            <Database className="w-4 h-4 text-cyan-400" />
            Binary Format Contract
          </div>
          <div className="space-y-2 text-xs font-mono">
            <div className="flex justify-between">
              <span className="text-neutral-400">Header Size:</span>
              <span className="text-neutral-200">64 bytes (checked)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Payload Alignment:</span>
              <span className="text-neutral-200">64-byte aligned</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Q4_0 Block Size:</span>
              <span className="text-neutral-200">32 weights / 18 bytes</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Block Layout:</span>
              <span className="text-neutral-200">FP16 scale + 16B nibbles</span>
            </div>
          </div>
        </div>

        {/* Loader & Memory Mapping */}
        <div className="terminal-panel p-4">
          <div className="flex items-center gap-2 text-xs font-mono text-neutral-300 font-semibold mb-3 border-b border-neutral-800 pb-2">
            <HardDrive className="w-4 h-4 text-purple-400" />
            Zero-Copy Loader
          </div>
          <div className="space-y-2 text-xs font-mono">
            <div className="flex justify-between">
              <span className="text-neutral-400">Load Latency:</span>
              <span className="text-emerald-400 font-semibold">{model.load_time_ms.toFixed(2)} ms</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Tensor Count:</span>
              <span className="text-neutral-200">{model.num_tensors} tensors</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Strategy:</span>
              <span className="text-neutral-200">{model.weight_loading_strategy}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Endianness:</span>
              <span className="text-neutral-200">Little-Endian (Checked)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tensor Directory Table */}
      <div className="terminal-panel flex-1 flex flex-col min-h-[400px]">
        {/* Table Controls */}
        <div className="p-4 border-b border-neutral-800 flex flex-col md:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-bold text-neutral-200 font-mono">
              Tensor Directory ({filteredTensors.length} / {tensors.length} entries)
            </h3>
          </div>

          <div className="flex items-center gap-3 w-full md:w-auto">
            {/* Filter by Dtype */}
            <div className="flex items-center gap-1.5 text-xs font-mono text-neutral-400">
              <Filter className="w-3.5 h-3.5" />
              <select
                value={dtypeFilter}
                onChange={(e) => setDtypeFilter(e.target.value)}
                className="bg-neutral-900 border border-neutral-800 text-neutral-200 rounded px-2 py-1 text-xs focus:outline-none focus:border-emerald-500"
              >
                <option value="ALL">All Types</option>
                {uniqueDtypes.map((dt) => (
                  <option key={dt} value={dt}>
                    {dt}
                  </option>
                ))}
              </select>
            </div>

            {/* Search Input */}
            <div className="relative flex-1 md:w-64">
              <Search className="w-3.5 h-3.5 text-neutral-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search tensors (e.g. layers.0.attn)..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-neutral-900 border border-neutral-800 text-neutral-200 text-xs rounded pl-8 pr-3 py-1.5 focus:outline-none focus:border-emerald-500 font-mono"
              />
            </div>
          </div>
        </div>

        {/* Scrollable Table */}
        <div className="overflow-x-auto overflow-y-auto flex-1 max-h-[480px]">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-neutral-900/80 sticky top-0 text-neutral-400 border-b border-neutral-800">
              <tr>
                <th className="py-2.5 px-3">#</th>
                <th className="py-2.5 px-3">Tensor Name</th>
                <th className="py-2.5 px-3">Dtype</th>
                <th className="py-2.5 px-3">Shape</th>
                <th className="py-2.5 px-3">Offset (Hex)</th>
                <th className="py-2.5 px-3 text-right">Size (Bytes)</th>
                <th className="py-2.5 px-3 text-right">Size (MiB)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-900">
              {filteredTensors.map((t, idx) => (
                <tr key={t.name || idx} className="hover:bg-neutral-900/50 transition-colors">
                  <td className="py-2 px-3 text-neutral-500">{idx + 1}</td>
                  <td className="py-2 px-3 text-emerald-400 font-semibold">{t.name}</td>
                  <td className="py-2 px-3">
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] ${
                        t.dtype === 'Q4_0'
                          ? 'bg-amber-950/60 text-amber-400 border border-amber-800/40'
                          : 'bg-cyan-950/60 text-cyan-400 border border-cyan-800/40'
                      }`}
                    >
                      {t.dtype}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-neutral-300">[{t.shape.join(', ')}]</td>
                  <td className="py-2 px-3 text-neutral-500">
                    0x{t.payload_offset.toString(16).toUpperCase()}
                  </td>
                  <td className="py-2 px-3 text-right text-neutral-300 font-mono">
                    {formatNumber(t.payload_nbytes)}
                  </td>
                  <td className="py-2 px-3 text-right text-neutral-400 font-mono">
                    {(t.payload_nbytes / (1024 * 1024)).toFixed(2)} MiB
                  </td>
                </tr>
              ))}

              {filteredTensors.length === 0 && (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-neutral-500">
                    No tensors match the specified query or filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
