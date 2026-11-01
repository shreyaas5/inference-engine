import {
  ChatMessage,
  GenerationConfig,
  GenerationMetrics,
  ModelMetadata,
  RuntimeTelemetry,
  AvailableModel,
  LogEntry,
  BenchmarkRecord,
} from '../types/index.js';

const API_BASE = '';

export interface SseHandlers {
  onStatusReport?: (data: {
    engine_state: string;
    model: ModelMetadata;
    runtime: RuntimeTelemetry;
    available_models: AvailableModel[];
  }) => void;
  onLog?: (entry: LogEntry) => void;
  onToken?: (tokenEv: {
    token_id: number;
    delta: string;
    generated_tokens: number;
    metrics: GenerationMetrics;
    request_id: string;
  }) => void;
  onGenerationStarted?: (ev: any) => void;
  onGenerationCompleted?: (ev: {
    request_id: string;
    text: string;
    stop_reason: string;
    metrics: GenerationMetrics;
  }) => void;
  onGenerationCancelled?: (ev: {
    request_id: string;
    text: string;
    metrics: GenerationMetrics;
  }) => void;
  onGenerationError?: (ev: { request_id: string; error: string }) => void;
  onContextCleared?: (ev: any) => void;
  onBenchmarkProgress?: (ev: any) => void;
  onBenchmarkCompleted?: (ev: { record: BenchmarkRecord }) => void;
  onEngineDisconnected?: (ev: { error: string }) => void;
}

export function subscribeToEvents(handlers: SseHandlers): () => void {
  const eventSource = new EventSource(`${API_BASE}/api/events`);

  eventSource.addEventListener('snapshot', (e) => {
    try {
      const data = JSON.parse(e.data);
      if (handlers.onStatusReport) {
        handlers.onStatusReport({
          engine_state: data.engine_state,
          model: data.model,
          runtime: data.runtime,
          available_models: data.available_models || [],
        });
      }
      if (data.logs && Array.isArray(data.logs) && handlers.onLog) {
        data.logs.forEach((l: LogEntry) => handlers.onLog!(l));
      }
    } catch {
      // ignore
    }
  });

  eventSource.addEventListener('status_report', (e) => {
    try {
      const data = JSON.parse(e.data);
      handlers.onStatusReport?.(data);
    } catch {}
  });

  eventSource.addEventListener('log', (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data.entry) handlers.onLog?.(data.entry);
    } catch {}
  });

  eventSource.addEventListener('generation_started', (e) => {
    try {
      handlers.onGenerationStarted?.(JSON.parse(e.data));
    } catch {}
  });

  eventSource.addEventListener('token', (e) => {
    try {
      handlers.onToken?.(JSON.parse(e.data));
    } catch {}
  });

  eventSource.addEventListener('generation_completed', (e) => {
    try {
      handlers.onGenerationCompleted?.(JSON.parse(e.data));
    } catch {}
  });

  eventSource.addEventListener('generation_cancelled', (e) => {
    try {
      handlers.onGenerationCancelled?.(JSON.parse(e.data));
    } catch {}
  });

  eventSource.addEventListener('generation_error', (e) => {
    try {
      handlers.onGenerationError?.(JSON.parse(e.data));
    } catch {}
  });

  eventSource.addEventListener('context_cleared', (e) => {
    try {
      handlers.onContextCleared?.(JSON.parse(e.data));
    } catch {}
  });

  eventSource.addEventListener('benchmark_progress', (e) => {
    try {
      handlers.onBenchmarkProgress?.(JSON.parse(e.data));
    } catch {}
  });

  eventSource.addEventListener('benchmark_completed', (e) => {
    try {
      handlers.onBenchmarkCompleted?.(JSON.parse(e.data));
    } catch {}
  });

  eventSource.addEventListener('engine_disconnected', (e) => {
    try {
      handlers.onEngineDisconnected?.(JSON.parse(e.data));
    } catch {}
  });

  return () => {
    eventSource.close();
  };
}

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.json();
}

export async function fetchModelInfo(): Promise<{
  model: ModelMetadata;
  available_models: AvailableModel[];
}> {
  const res = await fetch(`${API_BASE}/api/model`);
  return res.json();
}

export async function fetchSystemInfo(): Promise<{
  health: any;
  runtime: RuntimeTelemetry;
  model: ModelMetadata;
}> {
  const res = await fetch(`${API_BASE}/api/system`);
  return res.json();
}

export async function sendChatPrompt(
  messages: Array<{ role: string; content: string }>,
  config: GenerationConfig,
  requestId: string
) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      request_id: requestId,
      messages,
      temperature: config.temperature,
      top_p: config.top_p,
      top_k: config.top_k,
      max_tokens: config.max_tokens,
      seed: config.seed,
      threads: config.threads,
      backend: config.backend,
      model_path: config.model_path,
      stream: true,
    }),
  });
  return res;
}

export async function requestStopGeneration() {
  const res = await fetch(`${API_BASE}/api/chat/stop`, {
    method: 'POST',
  });
  return res.json();
}

export async function clearEngineContext() {
  const res = await fetch(`${API_BASE}/api/chat/clear`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request_id: `clear-${Date.now()}` }),
  });
  return res.json();
}

export async function reconfigureEngine(payload: {
  model_path?: string;
  threads?: number;
  backend?: string;
}) {
  const res = await fetch(`${API_BASE}/api/configure`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function startBenchmark(payload: {
  model_path?: string;
  threads?: number;
  backend?: string;
  runs?: number;
  max_tokens?: number;
  prompt?: string;
}) {
  const res = await fetch(`${API_BASE}/api/benchmark`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...payload, stream: true }),
  });
  return res;
}

export async function fetchBenchmarkHistory(): Promise<{ history: BenchmarkRecord[] }> {
  const res = await fetch(`${API_BASE}/api/benchmark/history`);
  return res.json();
}

export async function fetchLogs(): Promise<{ logs: LogEntry[] }> {
  const res = await fetch(`${API_BASE}/api/logs`);
  return res.json();
}

export async function clearLogs() {
  const res = await fetch(`${API_BASE}/api/logs/clear`, { method: 'POST' });
  return res.json();
}

// Convenient aliases for application components
export function subscribeSSE(onEvent: (event: any) => void): () => void {
  const eventSource = new EventSource(`${API_BASE}/api/events`);
  const eventTypes = [
    'snapshot',
    'status_report',
    'log',
    'generation_started',
    'token',
    'generation_completed',
    'generation_cancelled',
    'generation_error',
    'context_cleared',
    'benchmark_progress',
    'benchmark_completed',
    'engine_disconnected',
  ];

  eventTypes.forEach((type) => {
    eventSource.addEventListener(type, (e: MessageEvent) => {
      try {
        const parsed = JSON.parse(e.data);
        onEvent({ event: type, ...parsed });
      } catch {
        onEvent({ event: type });
      }
    });
  });

  eventSource.onerror = () => {
    // EventSource will auto-retry
  };

  return () => {
    eventSource.close();
  };
}

export async function sendChat(payload: {
  prompt: string;
  messages?: Array<{ role: string; content: string }>;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  threads?: number;
  stream?: boolean;
}) {
  const messagesToSend =
    payload.messages && payload.messages.length > 0
      ? payload.messages
      : [{ role: 'user', content: payload.prompt }];

  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages: messagesToSend,
      temperature: payload.temperature ?? 0.7,
      top_p: payload.top_p ?? 0.9,
      top_k: payload.top_k ?? 40,
      max_tokens: payload.max_tokens ?? 128,
      threads: payload.threads ?? 4,
      stream: payload.stream ?? true,
    }),
  });
  return res;
}

export const stopChat = requestStopGeneration;
export const clearChatContext = clearEngineContext;
export const configureEngine = reconfigureEngine;
export async function restartEngine() {
  return reconfigureEngine({});
}

