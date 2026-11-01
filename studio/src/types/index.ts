export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metrics?: GenerationMetrics;
  isStreaming?: boolean;
}

export interface GenerationMetrics {
  ttft_ms: number;
  prefill_ms: number;
  decode_ms?: number;
  total_ms: number;
  prompt_tokens: number;
  reused_kv_tokens: number;
  new_prompt_tokens: number;
  generated_tokens: number;
  prompt_tps: number;
  decode_tps: number;
  instant_tps?: number;
  current_rss_mib: number;
  peak_rss_mib: number;
  cpu_utilization_pct: number;
  kv_seq_len: number;
  max_seq_len: number;
  threads: number;
  backend: string;
}

export interface ModelTensorDesc {
  name: string;
  dtype: string;
  ndim: number;
  shape: number[];
  payload_offset: number;
  payload_nbytes: number;
}

export interface ModelMetadata {
  loaded: boolean;
  name: string;
  architecture: string;
  path: string;
  tokenizer_path: string;
  format: string;
  file_size_bytes: number;
  load_time_ms: number;
  is_mmap: boolean;
  weight_loading_strategy: string;
  num_tensors: number;
  parameter_count: number;
  config: {
    num_layers: number;
    hidden_dim: number;
    intermediate_dim: number;
    num_query_heads: number;
    num_kv_heads: number;
    head_dim: number;
    gqa_group_size: number;
    vocab_size: number;
    max_seq_len: number;
    rms_norm_eps: number;
    rope_theta: number;
  };
  tensors: ModelTensorDesc[];
}

export interface RuntimeTelemetry {
  active_backend: 'NEON' | 'AVX2' | 'Scalar' | string;
  best_backend: string;
  neon_available: boolean;
  avx2_available: boolean;
  scalar_available: boolean;
  num_threads: number;
  hardware_threads: number;
  kv_seq_len: number;
  kv_max_seq_len: number;
  kv_cache_size_mib: number;
  current_rss_mib: number;
  peak_rss_mib: number;
  os_arch: string;
  compiler: string;
}

export interface AvailableModel {
  path: string;
  format: string;
  size_bytes: number;
}

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

export interface GenerationConfig {
  temperature: number;
  top_p: number;
  top_k: number;
  max_tokens: number;
  seed: number;
  threads: number;
  backend: string;
  model_path: string;
}
