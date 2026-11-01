import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Square,
  Trash2,
  Sliders,
  ChevronDown,
  ChevronUp,
  Cpu,
  Zap,
  Sparkles,
  Bot,
  User,
  Copy,
  Check,
  RotateCcw,
} from 'lucide-react';
import { ChatMessage, GenerationConfig, RuntimeTelemetry } from '../types/index.js';

interface ChatPlaygroundProps {
  messages: ChatMessage[];
  isGenerating: boolean;
  onSendMessage: (content: string) => void;
  onStopGeneration: () => void;
  onClearContext: () => void;
  config: GenerationConfig;
  onChangeConfig: (newConfig: GenerationConfig) => void;
  runtime: RuntimeTelemetry | null;
}

export const ChatPlayground: React.FC<ChatPlaygroundProps> = ({
  messages,
  isGenerating,
  onSendMessage,
  onStopGeneration,
  onClearContext,
  config,
  onChangeConfig,
  runtime,
}) => {
  const [input, setInput] = useState('');
  const [showControls, setShowControls] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isGenerating]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || isGenerating) return;
    const prompt = input.trim();
    setInput('');
    onSendMessage(prompt);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const currentSeqLen = runtime?.kv_seq_len ?? 0;
  const maxSeqLen = runtime?.kv_max_seq_len ?? 2048;
  const contextPct = Math.min(100, (currentSeqLen / maxSeqLen) * 100);

  const SUGGESTED_PROMPTS = [
    'Explain how a CPU cache line works.',
    'What is the difference between SIMD and multithreading?',
    'Write a concise C++ function to compute dot product.',
    'Explain how causal self-attention works in TinyLlama.',
  ];

  return (
    <div className="flex-1 flex flex-col h-full bg-[#080b0f] relative font-mono overflow-hidden">
      {/* Top Bar: Context Window Bar & Quick Controls */}
      <div className="px-4 py-2 border-b border-[#1a222f] bg-[#0c1017] flex items-center justify-between text-xs text-slate-400">
        <div className="flex items-center gap-3">
          <span className="text-[11px] font-bold text-slate-300 flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-emerald-400" />
            <span>INFERENCE CONTEXT</span>
          </span>
          <div className="flex items-center gap-2">
            <div className="w-32 h-1.5 bg-[#171f2c] rounded-full overflow-hidden border border-[#232f42]">
              <div
                className={`h-full transition-all duration-300 ${
                  contextPct > 85 ? 'bg-amber-400' : 'bg-emerald-400'
                }`}
                style={{ width: `${contextPct}%` }}
              />
            </div>
            <span className="text-[10px] text-slate-400">
              {currentSeqLen} / {maxSeqLen} tok ({contextPct.toFixed(0)}%)
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowControls(!showControls)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] border transition cursor-pointer ${
              showControls
                ? 'bg-[#151d29] text-emerald-400 border-emerald-500/50'
                : 'bg-[#111620] text-slate-400 hover:text-slate-200 border-[#1f2937]'
            }`}
          >
            <Sliders className="w-3 h-3" />
            <span>Parameters</span>
            {showControls ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>
          <button
            onClick={onClearContext}
            title="Clear conversation history and KV cache"
            className="flex items-center gap-1 px-2.5 py-1 rounded text-[11px] bg-[#111620] hover:bg-red-950/30 text-slate-400 hover:text-red-400 border border-[#1f2937] hover:border-red-800/50 transition cursor-pointer"
          >
            <Trash2 className="w-3 h-3" />
            <span>Clear</span>
          </button>
        </div>
      </div>

      {/* Expandable Parameter Bar */}
      {showControls && (
        <div className="p-3 border-b border-[#1a222f] bg-[#0b0e14] grid grid-cols-2 md:grid-cols-5 gap-3 text-xs animate-in slide-in-from-top-2 duration-150">
          <div>
            <div className="flex justify-between text-slate-400 text-[10px] mb-1">
              <span>Temperature</span>
              <span className="text-emerald-400 font-bold">{config.temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="2"
              step="0.05"
              value={config.temperature}
              onChange={(e) => onChangeConfig({ ...config, temperature: parseFloat(e.target.value) })}
              className="w-full accent-emerald-500 cursor-pointer h-1.5 bg-[#1a2332] rounded"
            />
          </div>

          <div>
            <div className="flex justify-between text-slate-400 text-[10px] mb-1">
              <span>Top-P (Nucleus)</span>
              <span className="text-emerald-400 font-bold">{config.top_p.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={config.top_p}
              onChange={(e) => onChangeConfig({ ...config, top_p: parseFloat(e.target.value) })}
              className="w-full accent-emerald-500 cursor-pointer h-1.5 bg-[#1a2332] rounded"
            />
          </div>

          <div>
            <div className="flex justify-between text-slate-400 text-[10px] mb-1">
              <span>Top-K Filtering</span>
              <span className="text-emerald-400 font-bold">
                {config.top_k === 0 ? 'Off (0)' : config.top_k}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={config.top_k}
              onChange={(e) => onChangeConfig({ ...config, top_k: parseInt(e.target.value, 10) })}
              className="w-full accent-emerald-500 cursor-pointer h-1.5 bg-[#1a2332] rounded"
            />
          </div>

          <div>
            <div className="flex justify-between text-slate-400 text-[10px] mb-1">
              <span>Max New Tokens</span>
              <span className="text-cyan-400 font-bold">{config.max_tokens}</span>
            </div>
            <input
              type="range"
              min="16"
              max="512"
              step="16"
              value={config.max_tokens}
              onChange={(e) => onChangeConfig({ ...config, max_tokens: parseInt(e.target.value, 10) })}
              className="w-full accent-cyan-500 cursor-pointer h-1.5 bg-[#1a2332] rounded"
            />
          </div>

          <div>
            <div className="flex justify-between text-slate-400 text-[10px] mb-1">
              <span>Worker Threads</span>
              <span className="text-amber-400 font-bold">{config.threads}T</span>
            </div>
            <input
              type="range"
              min="1"
              max={runtime?.hardware_threads || 8}
              step="1"
              value={config.threads}
              onChange={(e) => onChangeConfig({ ...config, threads: parseInt(e.target.value, 10) })}
              className="w-full accent-amber-500 cursor-pointer h-1.5 bg-[#1a2332] rounded"
            />
          </div>
        </div>
      )}

      {/* Messages Stream Container */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-4">
            <div className="w-12 h-12 rounded-lg bg-[#0e141d] border border-[#10b981]/40 flex items-center justify-center text-[#10b981] shadow-[0_0_20px_rgba(16,185,129,0.15)]">
              <Sparkles className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider">
                BIOS V2 Inference Workstation
              </h3>
              <p className="text-xs text-slate-400 max-w-md mt-1">
                Pure C++17 inference runtime executing TinyLlama-1.1B on commodity CPU without external dependencies.
              </p>
            </div>

            {/* Prompt Starter Pills */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-w-xl w-full pt-2">
              {SUGGESTED_PROMPTS.map((promptText, idx) => (
                <button
                  key={idx}
                  onClick={() => onSendMessage(promptText)}
                  className="p-2.5 text-left bg-[#0e131b] hover:bg-[#141b26] border border-[#1c2635] hover:border-[#10b981]/50 rounded text-xs text-slate-300 transition-all cursor-pointer group"
                >
                  <span className="text-[#10b981] group-hover:underline block truncate">
                    &gt; {promptText}
                  </span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex flex-col space-y-1.5 ${
                msg.role === 'user' ? 'items-end' : 'items-start'
              }`}
            >
              <div className="flex items-center gap-2 text-[10px] text-slate-500 px-1">
                {msg.role === 'user' ? (
                  <>
                    <span>OPERATOR</span>
                    <User className="w-3 h-3 text-slate-400" />
                  </>
                ) : (
                  <>
                    <Bot className="w-3 h-3 text-emerald-400" />
                    <span className="text-emerald-400 font-bold">BIOS V2 (TINYLLAMA)</span>
                  </>
                )}
                <span>{new Date(msg.timestamp).toLocaleTimeString()}</span>
              </div>

              {/* Message Bubble */}
              <div
                className={`max-w-[85%] rounded-lg p-3 text-xs leading-relaxed border relative group ${
                  msg.role === 'user'
                    ? 'bg-[#151c27] border-[#222e40] text-slate-100 shadow-sm'
                    : 'bg-[#0d1219] border-[#182333] text-slate-200 shadow-[0_0_15px_rgba(0,0,0,0.5)]'
                }`}
              >
                <div className="whitespace-pre-wrap font-mono break-words">
                  {msg.content}
                  {msg.isStreaming && (
                    <span className="inline-block w-2 h-4 ml-1 bg-emerald-400 animate-pulse align-middle" />
                  )}
                </div>

                {/* Per-message Metrics Bar */}
                {msg.metrics && (
                  <div className="mt-2.5 pt-2 border-t border-[#1a2434] flex flex-wrap items-center gap-2 text-[10px] text-slate-400 font-mono">
                    <span className="bg-[#131b26] text-emerald-400 px-1.5 py-0.5 rounded border border-[#1e2a3c]">
                      {msg.metrics.decode_tps.toFixed(1)} tok/s
                    </span>
                    <span className="bg-[#131b26] text-cyan-400 px-1.5 py-0.5 rounded border border-[#1e2a3c]">
                      TTFT: {msg.metrics.ttft_ms.toFixed(0)} ms
                    </span>
                    <span className="bg-[#131b26] text-slate-300 px-1.5 py-0.5 rounded border border-[#1e2a3c]">
                      {msg.metrics.generated_tokens} tokens
                    </span>
                    {msg.metrics.reused_kv_tokens > 0 && (
                      <span className="bg-emerald-950/40 text-emerald-300 px-1.5 py-0.5 rounded border border-emerald-800/50">
                        KV Reused: {msg.metrics.reused_kv_tokens}
                      </span>
                    )}
                    <span className="text-slate-500">
                      ({(msg.metrics.total_ms / 1000).toFixed(2)}s)
                    </span>
                  </div>
                )}

                {/* Copy Button */}
                <button
                  onClick={() => handleCopy(msg.id, msg.content)}
                  className="absolute top-2 right-2 p-1 rounded bg-[#101722]/80 opacity-0 group-hover:opacity-100 transition text-slate-400 hover:text-slate-200 border border-[#202c3e]"
                  title="Copy text"
                >
                  {copiedId === msg.id ? (
                    <Check className="w-3 h-3 text-emerald-400" />
                  ) : (
                    <Copy className="w-3 h-3" />
                  )}
                </button>
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Bottom Composer Bar */}
      <div className="p-3 border-t border-[#1a222f] bg-[#0a0d13]">
        <form onSubmit={handleSubmit} className="relative flex flex-col gap-2">
          <div className="relative rounded-lg bg-[#0e131b] border border-[#1e293b] focus-within:border-emerald-500/60 focus-within:shadow-[0_0_12px_rgba(16,185,129,0.15)] transition-all">
            <textarea
              ref={textareaRef}
              rows={2}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                isGenerating
                  ? 'Generating response from TinyLlama C++ engine...'
                  : 'Enter prompt for TinyLlama (Press Enter to send, Shift+Enter for newline)...'
              }
              disabled={isGenerating}
              className="w-full p-2.5 text-xs text-slate-100 bg-transparent resize-none focus:outline-none placeholder:text-slate-600 font-mono"
            />

            <div className="flex items-center justify-between px-2.5 pb-2">
              <span className="text-[10px] text-slate-500">
                Format: <span className="text-slate-400">Zephyr Chat Template</span> | Backend:{' '}
                <span className="text-emerald-400 font-bold">{runtime?.active_backend || 'NEON'}</span>
              </span>

              <div className="flex items-center gap-2">
                {isGenerating ? (
                  <button
                    type="button"
                    onClick={onStopGeneration}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-red-950/40 hover:bg-red-900/60 border border-red-700/60 text-red-400 rounded text-xs font-mono font-bold transition active:scale-95 cursor-pointer shadow-[0_0_10px_rgba(239,68,68,0.2)]"
                  >
                    <Square className="w-3 h-3 fill-current" />
                    <span>STOP</span>
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!input.trim()}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-[#10b981] hover:bg-[#059669] disabled:bg-[#1a2332] text-[#080b0f] disabled:text-slate-600 rounded text-xs font-mono font-bold transition active:scale-95 cursor-pointer disabled:cursor-not-allowed shadow-[0_0_12px_rgba(16,185,129,0.25)]"
                  >
                    <span>EXECUTE</span>
                    <Send className="w-3 h-3" />
                  </button>
                )}
              </div>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};
