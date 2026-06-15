// Typed client for the webapp2 backend.

export type Databases = Record<string, Record<string, string[]>>;

export type Model = { path: string; is_moe: boolean };

export type OpRow = { op: string; ms: number; source: string };
export type MemRow = { name: string; gib: number };
export type Card = { label: string; value: number; unit: string };
export type Phase = { name: string; rows: OpRow[] };
export type MemGroup = { label: string; rows: MemRow[]; kv_bytes_per_token: number | null };

export type EstimateResponse = {
  mode: string;
  cards: Card[];
  phases: Phase[];
  memory_groups: MemGroup[];
  meta: Record<string, string | number>;
};

export type EstimateRequest = {
  mode: string;
  model_path: string;
  system_name: string;
  backend_name: string;
  backend_version: string | null;
  database_mode: string;
  isl: number;
  osl: number;
  // static / agg
  batch_size?: number;
  tp_size?: number;
  pp_size?: number;
  attention_dp_size?: number;
  moe_tp_size?: number | null;
  moe_ep_size?: number | null;
  // disagg prefill
  prefill_tp_size?: number;
  prefill_attention_dp_size?: number;
  prefill_moe_tp_size?: number | null;
  prefill_moe_ep_size?: number | null;
  prefill_batch_size?: number;
  prefill_num_workers?: number;
  // disagg decode
  decode_tp_size?: number;
  decode_attention_dp_size?: number;
  decode_moe_tp_size?: number | null;
  decode_moe_ep_size?: number | null;
  decode_batch_size?: number;
  decode_num_workers?: number;
};

export async function fetchDatabases(): Promise<Databases> {
  const r = await fetch("/api/databases");
  if (!r.ok) throw new Error(`databases: ${r.status}`);
  return r.json();
}

export async function fetchModels(): Promise<Model[]> {
  const r = await fetch("/api/models");
  if (!r.ok) throw new Error(`models: ${r.status}`);
  return r.json();
}

export async function postEstimate(req: EstimateRequest): Promise<EstimateResponse> {
  const r = await fetch("/api/estimate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail || `estimate failed: ${r.status}`);
  }
  return r.json();
}

// --- Config ---

export type ConfigStatus = {
  provider: string;
  model: string;
  base_url: string;
  key_set: boolean;
  provider_models: Record<string, string[]>;
};

export async function fetchConfig(): Promise<ConfigStatus> {
  const r = await fetch("/api/config");
  if (!r.ok) throw new Error(`config: ${r.status}`);
  return r.json();
}

export async function saveConfig(
  provider: string,
  model: string,
  apiKey: string | null,
  baseUrl: string
): Promise<ConfigStatus> {
  const r = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, model, api_key: apiKey, base_url: baseUrl }),
  });
  if (!r.ok) throw new Error(`save config: ${r.status}`);
  return r.json();
}

// --- LLM task generation ---

export type GenerateResult = { yaml: string; valid: boolean; error: string | null; attempts: number };

export async function generateTask(prompt: string): Promise<GenerateResult> {
  const r = await fetch("/api/llm/generate-task", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail || `generate failed: ${r.status}`);
  }
  return r.json();
}

// --- Sweep ---

export type SweepPoint = { x: number; y: number; on_pareto: boolean; label: string };
export type SweepResult = {
  empty: boolean;
  x_col?: string;
  y_col?: string;
  n_feasible?: number;
  n_pareto?: number;
  columns?: string[];
  points?: SweepPoint[];
  pareto?: Record<string, number>[];
};
export type SweepJob = { status: string; result: SweepResult | null; error: string | null };

async function postSweep(body: Record<string, unknown>): Promise<{ job_id: string }> {
  const r = await fetch("/api/sweep", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `sweep failed: ${r.status}`);
  }
  return r.json();
}

export function startSweep(task: Record<string, unknown>): Promise<{ job_id: string }> {
  return postSweep({ task });
}

export function startSweepYaml(yaml: string): Promise<{ job_id: string }> {
  return postSweep({ yaml });
}

export async function getSweep(jobId: string): Promise<SweepJob> {
  const r = await fetch(`/api/sweep/${jobId}`);
  if (!r.ok) throw new Error(`sweep status: ${r.status}`);
  return r.json();
}
