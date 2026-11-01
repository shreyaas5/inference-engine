import { spawn, ChildProcessWithoutNullStreams } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as readline from 'node:readline';
import { EventEmitter } from 'node:events';

export interface LogEntry {
  id: string;
  timestamp: string;
  level: 'INFO' | 'WARN' | 'ERROR';
  source: 'cpp-engine' | 'studio-api';
  message: string;
}

export interface BenchmarkStatGroup {
  mean: number;
  median: number;
  min: number;
  max: number;
  stddev: number;
}

export interface BenchmarkRecord {
  id: string;
  timestamp: string;
  label: string;
  is_historical: boolean;
  model_path: string;
  format: string;
  backend: string;
  threads: number;
  runs: number;
  model_load_ms: number;
  prompt: string;
  sample_output?: string;
  prompt_tokens: number;
  generated_tokens: number;
  current_rss_mib: number;
  peak_rss_mib: number;
  cpu_utilization_mean_pct: number;
  os_arch: string;
  compiler: string;
  ttft_ms: BenchmarkStatGroup;
  prompt_tps: BenchmarkStatGroup;
  decode_tps: BenchmarkStatGroup;
  individual_runs?: Array<{
    run_index: number;
    ttft_ms: number;
    prefill_ms: number;
    decode_ms: number;
    total_ms: number;
    prompt_tps: number;
    decode_tps: number;
    cpu_utilization_pct: number;
    generated_tokens: number;
  }>;
}

const HISTORICAL_BENCHMARKS: BenchmarkRecord[] = [
  {
    id: 'hist-m6-q4-4t',
    timestamp: '2026-11-01T13:10:00+05:30',
    label: 'M6 Historical Baseline (Q4_0, 4 Threads, 3 Runs)',
    is_historical: true,
    model_path: 'models/tinyllama-1.1b-q4_0.ie',
    format: 'Q4_0',
    backend: 'NEON',
    threads: 4,
    runs: 3,
    model_load_ms: 3.742,
    prompt: 'Explain entropy.',
    prompt_tokens: 20,
    generated_tokens: 32,
    current_rss_mib: 658.2,
    peak_rss_mib: 658.2,
    cpu_utilization_mean_pct: 368.0,
    os_arch: 'macOS ARM64 (Apple Silicon)',
    compiler: 'AppleClang 17.0.0 (-O3)',
    ttft_ms: { mean: 561.259, median: 556.301, min: 555.443, max: 572.033, stddev: 7.624 },
    prompt_tps: { mean: 35.648, median: 35.96, min: 34.97, max: 36.015, stddev: 0.48 },
    decode_tps: { mean: 35.255, median: 35.292, min: 35.055, max: 35.418, stddev: 0.15 },
  },
  {
    id: 'hist-m7-q4-4t',
    timestamp: '2026-11-01T13:47:37+05:30',
    label: 'M7 Clean-Checkout Verification (Q4_0, 4 Threads, 5 Runs)',
    is_historical: true,
    model_path: 'models/tinyllama-1.1b-q4_0.ie',
    format: 'Q4_0',
    backend: 'NEON',
    threads: 4,
    runs: 5,
    model_load_ms: 1.323,
    prompt: 'Explain entropy.',
    prompt_tokens: 20,
    generated_tokens: 32,
    current_rss_mib: 658.16,
    peak_rss_mib: 658.16,
    cpu_utilization_mean_pct: 355.0,
    os_arch: 'macOS ARM64 (Apple Silicon)',
    compiler: 'AppleClang 17.0.0 (-O3)',
    ttft_ms: { mean: 657.792, median: 611.197, min: 579.89, max: 885.817, stddev: 114.8 },
    prompt_tps: { mean: 31.171, median: 32.729, min: 22.581, max: 34.496, stddev: 4.35 },
    decode_tps: { mean: 34.032, median: 34.373, min: 33.215, max: 34.631, stddev: 0.56 },
  },
  {
    id: 'hist-m5-fp32-4t',
    timestamp: '2026-11-01T06:18:00+05:30',
    label: 'M5 Historical Reference (FP32 Unquantised, 4 Threads)',
    is_historical: true,
    model_path: 'models/tinyllama-1.1b-fp32.ie',
    format: 'FP32',
    backend: 'NEON',
    threads: 4,
    runs: 1,
    model_load_ms: 4.82,
    prompt: 'Explain how a CPU executes an instruction.',
    prompt_tokens: 21,
    generated_tokens: 64,
    current_rss_mib: 3956.6,
    peak_rss_mib: 3956.6,
    cpu_utilization_mean_pct: 372.0,
    os_arch: 'macOS ARM64 (Apple Silicon)',
    compiler: 'AppleClang 17.0.0 (-O3)',
    ttft_ms: { mean: 6096.8, median: 6096.8, min: 6096.8, max: 6096.8, stddev: 0.0 },
    prompt_tps: { mean: 3.44, median: 3.44, min: 3.44, max: 3.44, stddev: 0.0 },
    decode_tps: { mean: 16.1, median: 16.1, min: 16.1, max: 16.1, stddev: 0.0 },
  },
];

export class EngineManager extends EventEmitter {
  private repoRoot: string;
  private binaryPath: string;
  private worker: ChildProcessWithoutNullStreams | null = null;
  private startTime: number = Date.now();
  private logCounter = 0;

  public engineConnected = false;
  public engineState: 'starting' | 'ready' | 'generating' | 'benchmarking' | 'error' | 'disconnected' = 'starting';
  public lastError = '';
  public modelInfo: any = null;
  public runtimeInfo: any = null;
  public availableModels: Array<{ path: string; format: string; size_bytes: number }> = [];
  public logs: LogEntry[] = [];
  public benchmarkHistory: BenchmarkRecord[] = [...HISTORICAL_BENCHMARKS];

  private pendingRequests = new Map<
    string,
    {
      resolve: (val: any) => void;
      reject: (err: Error) => void;
      onEvent?: (ev: any) => void;
    }
  >();

  constructor() {
    super();
    this.setMaxListeners(100);
    this.repoRoot = path.resolve(import.meta.dirname, '../..');
    const exeName = process.platform === 'win32' ? 'inference-engine.exe' : 'inference-engine';
    this.binaryPath = path.join(this.repoRoot, 'build', exeName);

    this.setupProcessCleanup();
    this.startWorker();
  }

  private setupProcessCleanup() {
    const cleanup = () => {
      if (this.worker && !this.worker.killed) {
        try {
          this.worker.stdin.write('{"cmd":"shutdown"}\n');
          this.worker.kill('SIGTERM');
        } catch {
          // ignore
        }
      }
    };
    process.on('exit', cleanup);
    process.on('SIGINT', () => {
      cleanup();
      process.exit(0);
    });
    process.on('SIGTERM', () => {
      cleanup();
      process.exit(0);
    });
  }

  public addLog(level: 'INFO' | 'WARN' | 'ERROR', source: 'cpp-engine' | 'studio-api', message: string) {
    const entry: LogEntry = {
      id: `log-${++this.logCounter}-${Date.now()}`,
      timestamp: new Date().toISOString(),
      level,
      source,
      message,
    };
    this.logs.push(entry);
    if (this.logs.length > 500) {
      this.logs.shift();
    }
    this.emit('sse_event', { event: 'log', entry });
  }

  public clearLogs() {
    this.logs = [];
    this.addLog('INFO', 'studio-api', 'Runtime logs cleared by operator.');
  }

  public startWorker() {
    if (this.worker && !this.worker.killed) {
      return;
    }

    if (!fs.existsSync(this.binaryPath)) {
      this.engineConnected = false;
      this.engineState = 'error';
      this.lastError = `C++ binary not found at ${this.binaryPath}. Run 'cmake --build build' first.`;
      this.addLog('ERROR', 'studio-api', this.lastError);
      return;
    }

    this.engineState = 'starting';
    this.addLog('INFO', 'studio-api', `Spawning persistent C++ engine process: ${this.binaryPath} --studio-worker`);

    const child = spawn(this.binaryPath, ['--studio-worker'], {
      cwd: this.repoRoot,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    this.worker = child;
    this.engineConnected = true;

    const rl = readline.createInterface({ input: child.stdout });
    rl.on('line', (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      try {
        const ev = JSON.parse(trimmed);
        this.handleEngineEvent(ev);
      } catch (err) {
        this.addLog('WARN', 'cpp-engine', `Raw stdout: ${trimmed}`);
      }
    });

    const errRl = readline.createInterface({ input: child.stderr });
    errRl.on('line', (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      this.addLog('WARN', 'cpp-engine', trimmed);
    });

    child.on('exit', (code, signal) => {
      this.engineConnected = false;
      this.engineState = 'disconnected';
      this.worker = null;
      const msg = `C++ engine process exited (code=${code}, signal=${signal})`;
      this.lastError = msg;
      this.addLog('ERROR', 'studio-api', msg);
      this.emit('sse_event', { event: 'engine_disconnected', error: msg });

      for (const [, req] of this.pendingRequests) {
        req.reject(new Error(msg));
      }
      this.pendingRequests.clear();
    });
  }

  private handleEngineEvent(ev: any) {
    const type = ev.event;
    const reqId = ev.request_id || '';

    if (type === 'log') {
      this.addLog(ev.level || 'INFO', 'cpp-engine', ev.message || '');
      return;
    }

    if (type === 'engine_ready' || type === 'status_report') {
      this.engineConnected = true;
      this.engineState = ev.engine_state || 'ready';
      this.lastError = ev.last_error || '';
      if (ev.model) this.modelInfo = ev.model;
      if (ev.runtime) this.runtimeInfo = ev.runtime;
      if (ev.available_models) this.availableModels = ev.available_models;

      this.emit('sse_event', {
        event: 'status_report',
        engine_state: this.engineState,
        model: this.modelInfo,
        runtime: this.runtimeInfo,
        available_models: this.availableModels,
      });

      if (reqId && this.pendingRequests.has(reqId)) {
        const pending = this.pendingRequests.get(reqId)!;
        this.pendingRequests.delete(reqId);
        pending.resolve(ev);
      }
      return;
    }

    if (type === 'generation_started') {
      this.engineState = 'generating';
      if (this.runtimeInfo && typeof ev.kv_seq_len === 'number') {
        this.runtimeInfo.kv_seq_len = ev.kv_seq_len;
      }
      this.emit('sse_event', ev);
      if (reqId && this.pendingRequests.has(reqId)) {
        this.pendingRequests.get(reqId)!.onEvent?.(ev);
      }
      return;
    }

    if (type === 'token') {
      if (this.runtimeInfo && ev.metrics) {
        this.runtimeInfo.kv_seq_len = ev.metrics.kv_seq_len;
        this.runtimeInfo.current_rss_mib = ev.metrics.current_rss_mib;
        this.runtimeInfo.peak_rss_mib = ev.metrics.peak_rss_mib;
      }
      this.emit('sse_event', ev);
      if (reqId && this.pendingRequests.has(reqId)) {
        this.pendingRequests.get(reqId)!.onEvent?.(ev);
      }
      return;
    }

    if (type === 'generation_completed' || type === 'generation_cancelled') {
      this.engineState = 'ready';
      if (this.runtimeInfo && ev.metrics) {
        this.runtimeInfo.kv_seq_len = ev.metrics.kv_seq_len;
        this.runtimeInfo.current_rss_mib = ev.metrics.current_rss_mib;
        this.runtimeInfo.peak_rss_mib = ev.metrics.peak_rss_mib;
      }
      this.emit('sse_event', ev);
      if (reqId && this.pendingRequests.has(reqId)) {
        const pending = this.pendingRequests.get(reqId)!;
        pending.onEvent?.(ev);
        this.pendingRequests.delete(reqId);
        pending.resolve(ev);
      }
      return;
    }

    if (type === 'generation_error') {
      this.engineState = this.modelInfo?.loaded ? 'ready' : 'error';
      this.lastError = ev.error || 'Unknown engine error';
      this.emit('sse_event', ev);
      if (reqId && this.pendingRequests.has(reqId)) {
        const pending = this.pendingRequests.get(reqId)!;
        pending.onEvent?.(ev);
        this.pendingRequests.delete(reqId);
        pending.reject(new Error(ev.error || 'Engine error'));
      }
      return;
    }

    if (type === 'context_cleared') {
      if (this.runtimeInfo) {
        this.runtimeInfo.kv_seq_len = 0;
        if (typeof ev.current_rss_mib === 'number') {
          this.runtimeInfo.current_rss_mib = ev.current_rss_mib;
        }
        if (typeof ev.peak_rss_mib === 'number') {
          this.runtimeInfo.peak_rss_mib = ev.peak_rss_mib;
        }
      }
      this.emit('sse_event', ev);
      if (reqId && this.pendingRequests.has(reqId)) {
        const pending = this.pendingRequests.get(reqId)!;
        this.pendingRequests.delete(reqId);
        pending.resolve(ev);
      }
      return;
    }

    if (type === 'benchmark_started' || type === 'benchmark_progress') {
      this.engineState = 'benchmarking';
      this.emit('sse_event', ev);
      if (reqId && this.pendingRequests.has(reqId)) {
        this.pendingRequests.get(reqId)!.onEvent?.(ev);
      }
      return;
    }

    if (type === 'benchmark_completed') {
      this.engineState = 'ready';
      if (this.runtimeInfo) {
        this.runtimeInfo.kv_seq_len = 0;
      }
      const res = ev.result;
      const record: BenchmarkRecord = {
        id: `bench-${Date.now()}`,
        timestamp: new Date().toISOString(),
        label: `Live Benchmark (${res.format}, ${res.threads}T, ${res.runs} ${res.runs === 1 ? 'Run' : 'Runs'})`,
        is_historical: false,
        ...res,
      };
      this.benchmarkHistory.unshift(record);
      this.emit('sse_event', { event: 'benchmark_completed', request_id: reqId, record });

      if (reqId && this.pendingRequests.has(reqId)) {
        const pending = this.pendingRequests.get(reqId)!;
        pending.onEvent?.({ event: 'benchmark_completed', request_id: reqId, record });
        this.pendingRequests.delete(reqId);
        pending.resolve(record);
      }
      return;
    }
  }

  public sendCommand(
    cmdObj: Record<string, any>,
    onEvent?: (ev: any) => void
  ): Promise<any> {
    if (!this.worker || this.worker.killed || !this.engineConnected) {
      this.startWorker();
      if (!this.worker || !this.engineConnected) {
        return Promise.reject(new Error(this.lastError || 'C++ engine worker is not running'));
      }
    }

    const requestId = cmdObj.request_id || `req-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    const payload = JSON.stringify({ ...cmdObj, request_id: requestId });

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(requestId, { resolve, reject, onEvent });
      try {
        this.worker!.stdin.write(payload + '\n');
      } catch (err: any) {
        this.pendingRequests.delete(requestId);
        reject(new Error(`Failed to write to C++ worker stdin: ${err?.message || err}`));
      }
    });
  }

  public requestStop() {
    if (this.worker && !this.worker.killed && this.engineConnected) {
      this.addLog('WARN', 'studio-api', 'Sending stop signal to C++ engine worker.');
      this.worker.stdin.write('{"cmd":"stop"}\n');
      return true;
    }
    return false;
  }

  public getHealth() {
    return {
      status: this.engineConnected ? 'ok' : 'error',
      engine_connected: this.engineConnected,
      engine_state: this.engineState,
      pid: this.worker?.pid || null,
      uptime_sec: Math.floor((Date.now() - this.startTime) / 1000),
      model_loaded: Boolean(this.modelInfo?.loaded),
      last_error: this.lastError,
    };
  }
}

let globalEngineManager: EngineManager | null = null;
export function getEngineManager(): EngineManager {
  if (!globalEngineManager) {
    globalEngineManager = new EngineManager();
  }
  return globalEngineManager;
}

