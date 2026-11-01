import React, { useState, useEffect, useRef } from 'react';
import {
  ChatMessage,
  GenerationConfig,
  GenerationMetrics,
  ModelMetadata,
  RuntimeTelemetry,
  AvailableModel,
  LogEntry,
  BenchmarkRecord,
} from './types/index.js';
import {
  subscribeSSE,
  sendChat,
  stopChat,
  clearChatContext,
  configureEngine,
  restartEngine,
  clearLogs,
} from './api/client.js';
import { NavigationSidebar } from './components/NavigationSidebar.js';
import { ChatPlayground } from './components/ChatPlayground.js';
import { ObservabilityPanel } from './components/ObservabilityPanel.js';
import { BenchmarkLab } from './components/BenchmarkLab.js';
import { ModelInspector } from './components/ModelInspector.js';
import { RuntimeLogViewer } from './components/RuntimeLogViewer.js';
import { SettingsModal } from './components/SettingsModal.js';

export const App: React.FC = () => {
  // Navigation & View State
  const [currentTab, setCurrentTab] = useState<'chat' | 'benchmark' | 'inspector' | 'logs'>('chat');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Engine & Model State
  const [engineConnected, setEngineConnected] = useState(false);
  const [engineState, setEngineState] = useState<string>('disconnected');
  const [model, setModel] = useState<ModelMetadata | null>(null);
  const [runtime, setRuntime] = useState<RuntimeTelemetry | null>(null);
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);

  // Generation & Chat State
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [latestMetrics, setLatestMetrics] = useState<GenerationMetrics | null>(null);
  const [tpsHistory, setTpsHistory] = useState<number[]>([35.2, 35.5, 34.8, 35.3, 35.1]);

  // Benchmark State
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [benchmarkProgress, setBenchmarkProgress] = useState<{
    completed: number;
    total: number;
    latest?: any;
  } | null>(null);

  // Logs State
  const [logs, setLogs] = useState<LogEntry[]>([]);

  // Generation Parameters
  const [genConfig, setGenConfig] = useState<GenerationConfig>({
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    max_tokens: 128,
    seed: 42,
    threads: 4,
    backend: 'NEON',
    model_path: 'models/tinyllama-1.1b-q4_0.ie',
  });

  // Keep a reference to current messages for SSE updates
  const activeMessageIdRef = useRef<string | null>(null);

  // Subscribe to persistent SSE events from Studio backend
  useEffect(() => {
    const unsubscribe = subscribeSSE((event: any) => {
      const type = event.event;

      if (type === 'snapshot') {
        setEngineConnected(event.engine_connected);
        setEngineState(event.engine_state);
        if (event.model) setModel(event.model);
        if (event.runtime) {
          setRuntime(event.runtime);
          if (event.runtime.num_threads) {
            setGenConfig((prev) => ({ ...prev, threads: event.runtime.num_threads }));
          }
        }
        if (event.available_models) setAvailableModels(event.available_models);
        if (event.logs) setLogs(event.logs);
      } else if (type === 'status_report') {
        setEngineConnected(true);
        setEngineState(event.engine_state);
        if (event.model) setModel(event.model);
        if (event.runtime) setRuntime(event.runtime);
        if (event.available_models) setAvailableModels(event.available_models);
      } else if (type === 'engine_disconnected') {
        setEngineConnected(false);
        setEngineState('disconnected');
      } else if (type === 'log') {
        if (event.entry) {
          setLogs((prev) => [...prev.slice(-499), event.entry]);
        }
      } else if (type === 'generation_started') {
        setIsGenerating(true);
      } else if (type === 'token') {
        const text = event.text || '';
        const tokenMetrics = event.metrics as GenerationMetrics;
        if (tokenMetrics) {
          setLatestMetrics(tokenMetrics);
          if (tokenMetrics.instant_tps !== undefined && tokenMetrics.instant_tps > 0) {
            setTpsHistory((prev) => {
              const next = [...prev, tokenMetrics.instant_tps!];
              return next.length > 30 ? next.slice(next.length - 30) : next;
            });
          }
        }

        setMessages((prev) => {
          if (!activeMessageIdRef.current) return prev;
          return prev.map((msg) => {
            if (msg.id === activeMessageIdRef.current) {
              return {
                ...msg,
                content: msg.content + text,
                isStreaming: true,
                metrics: tokenMetrics || msg.metrics,
              };
            }
            return msg;
          });
        });
      } else if (type === 'generation_completed' || type === 'generation_cancelled') {
        setIsGenerating(false);
        const finalMetrics = event.metrics as GenerationMetrics;
        if (finalMetrics) {
          setLatestMetrics(finalMetrics);
        }

        setMessages((prev) => {
          if (!activeMessageIdRef.current) return prev;
          return prev.map((msg) => {
            if (msg.id === activeMessageIdRef.current) {
              return {
                ...msg,
                isStreaming: false,
                metrics: finalMetrics || msg.metrics,
              };
            }
            return msg;
          });
        });
        activeMessageIdRef.current = null;
      } else if (type === 'generation_error') {
        setIsGenerating(false);
        const errMsg = event.error || 'Engine generation error occurred';
        setMessages((prev) => {
          if (!activeMessageIdRef.current) return prev;
          return prev.map((msg) => {
            if (msg.id === activeMessageIdRef.current) {
              return {
                ...msg,
                content: msg.content ? `${msg.content}\n\n[Error: ${errMsg}]` : `[Error: ${errMsg}]`,
                isStreaming: false,
              };
            }
            return msg;
          });
        });
        activeMessageIdRef.current = null;
      } else if (type === 'benchmark_progress') {
        setIsBenchmarking(true);
        setBenchmarkProgress({
          completed: event.completed,
          total: event.total,
          latest: event.latest,
        });
      } else if (type === 'benchmark_completed') {
        setIsBenchmarking(false);
        setBenchmarkProgress(null);
      }
    });

    return () => {
      unsubscribe();
    };
  }, []);

  // Send a new prompt to the persistent C++ worker
  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isGenerating) return;

    const userMsgId = `user-${Date.now()}`;
    const assistantMsgId = `assistant-${Date.now() + 1}`;
    activeMessageIdRef.current = assistantMsgId;

    const userMsg: ChatMessage = {
      id: userMsgId,
      role: 'user',
      content: content.trim(),
      timestamp: new Date().toISOString(),
    };

    const assistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsGenerating(true);

    try {
      await sendChat({
        prompt: content.trim(),
        temperature: genConfig.temperature,
        top_p: genConfig.top_p,
        top_k: genConfig.top_k,
        max_tokens: genConfig.max_tokens,
        threads: genConfig.threads,
        stream: true,
      });
    } catch (err: any) {
      setIsGenerating(false);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMsgId
            ? { ...msg, content: `Failed to initiate inference: ${err.message}`, isStreaming: false }
            : msg
        )
      );
      activeMessageIdRef.current = null;
    }
  };

  const handleStopGeneration = async () => {
    try {
      await stopChat();
    } catch (err) {
      console.error('Failed to stop generation:', err);
    }
  };

  const handleClearContext = async () => {
    try {
      await clearChatContext();
      setMessages([]);
      activeMessageIdRef.current = null;
      setLatestMetrics(null);
      if (runtime) {
        setRuntime({ ...runtime, kv_seq_len: 0 });
      }
    } catch (err) {
      console.error('Failed to clear context:', err);
    }
  };

  const handleConfigure = async (cfg: {
    model_path?: string;
    threads?: number;
    backend?: string;
  }) => {
    await configureEngine(cfg);
  };

  const handleReload = async () => {
    await restartEngine();
  };

  const handleClearLogs = async () => {
    await clearLogs();
  };

  const handleBenchmarkDone = (record: BenchmarkRecord) => {
    setLatestMetrics({
      ttft_ms: record.ttft_ms ? record.ttft_ms.mean : 0,
      prefill_ms: 0,
      total_ms: 0,
      prompt_tokens: record.prompt_tokens || 0,
      reused_kv_tokens: 0,
      new_prompt_tokens: record.prompt_tokens || 0,
      generated_tokens: record.generated_tokens || 0,
      prompt_tps: record.prompt_tps ? record.prompt_tps.mean : 0,
      decode_tps: record.decode_tps ? record.decode_tps.mean : 0,
      current_rss_mib: record.current_rss_mib || 0,
      peak_rss_mib: record.peak_rss_mib || 0,
      cpu_utilization_pct: record.cpu_utilization_mean_pct || 0,
      kv_seq_len: 0,
      max_seq_len: 2048,
      threads: record.threads || 4,
      backend: record.backend || 'NEON',
    });
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#080a0d] text-neutral-200 antialiased select-none">
      {/* 1. Left Navigation Panel */}
      <NavigationSidebar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        onNewChat={handleClearContext}
        onOpenSettings={() => setIsSettingsOpen(true)}
        engineConnected={engineConnected}
        engineState={engineState}
        model={model}
        runtime={runtime}
        chatTurnCount={messages.filter((m) => m.role === 'user').length}
        onClearContext={handleClearContext}
      />

      {/* 2. Centre Main Workstation Area */}
      <main className="flex-1 flex flex-col min-w-0 h-full overflow-hidden border-r border-neutral-800">
        {currentTab === 'chat' && (
          <ChatPlayground
            messages={messages}
            isGenerating={isGenerating}
            onSendMessage={handleSendMessage}
            onStopGeneration={handleStopGeneration}
            onClearContext={handleClearContext}
            config={genConfig}
            onChangeConfig={setGenConfig}
            runtime={runtime}
          />
        )}

        {currentTab === 'benchmark' && (
          <BenchmarkLab
            model={model}
            runtime={runtime}
            availableModels={availableModels}
            isBenchmarking={isBenchmarking}
            benchmarkProgress={benchmarkProgress}
            onBenchmarkDone={handleBenchmarkDone}
          />
        )}

        {currentTab === 'inspector' && <ModelInspector model={model} />}

        {currentTab === 'logs' && (
          <RuntimeLogViewer logs={logs} onClearLogs={handleClearLogs} />
        )}
      </main>

      {/* 3. Right Observability Panel (Telemetry, Live Gauges, KV Monitor) */}
      <aside className="w-80 2xl:w-96 h-full flex flex-col shrink-0 bg-[#07090d] hidden lg:flex">
        <ObservabilityPanel
          metrics={latestMetrics}
          runtime={runtime}
          model={model}
          instantTpsHistory={tpsHistory}
          isGenerating={isGenerating}
        />
      </aside>

      {/* 4. Configuration & Runtime Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        model={model}
        runtime={runtime}
        availableModels={availableModels}
        onConfigure={handleConfigure}
        onReload={handleReload}
      />
    </div>
  );
};
