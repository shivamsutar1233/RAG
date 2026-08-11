/**
 * Typed client for the FastAPI backend.
 *
 * In production the Next export is served by FastAPI itself, so same-origin
 * works. In `next dev` the app runs on :3000 while the API is on :8000, hence
 * the env override — the backend already allows all CORS origins.
 */
import { accessToken } from "./supabase";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8000"
    : "");

export interface StagedFile {
  name: string;
  size: string;
  status: string;
}

export interface StatusResponse {
  status: string;
  database_loaded: boolean;
  document_chunks: number;
  routing_method: string;
  reranker_provider: string;
  langsmith_tracing: boolean;
  staged_files: StagedFile[];
  workspace?: string;
  user_email?: string | null;
  auth_enabled?: boolean;
  storage_enabled?: boolean;
  llm_provider?: string;
  llm_model?: string;
  llm_key_configured?: boolean;
  embedding_provider?: string;
  embedding_model?: string;
  embedding_key_configured?: boolean;
  eval_llm_provider?: string;
  eval_llm_model?: string;
  eval_key_configured?: boolean;
  error?: string;
}

export interface ProviderEntry {
  id: string;
  default_model: string;
  key_env: string | null;
  requires_key: boolean;
  key_configured: boolean;
}

export interface ProviderCatalog {
  chat: ProviderEntry[];
  embedding: ProviderEntry[];
  chat_only: string[];
}

export interface SourceDocument {
  title: string;
  source: string;
  page: number | null;
  snippet: string;
}

export interface BuildMeta {
  id: string;
  status: "running" | "completed" | "failed";
  started_at: string;
  completed_at: string | null;
  raptor: boolean;
  error: string | null;
}

/** Metric keys the backend can report. Reference-based ones are absent when the
 *  test set has no ground-truth answers. */
export type EvalMetric =
  | "faithfulness"
  | "answer_relevancy"
  | "context_precision"
  | "context_recall"
  | "factual_correctness";

export const METRIC_LABELS: Record<EvalMetric, string> = {
  faithfulness: "Faithfulness",
  answer_relevancy: "Answer relevancy",
  context_precision: "Context precision",
  context_recall: "Context recall",
  factual_correctness: "Factual correctness",
};

export const METRIC_HELP: Record<EvalMetric, string> = {
  faithfulness: "Is the answer actually supported by the retrieved text? Low means hallucination.",
  answer_relevancy: "Does the answer address the question that was asked?",
  context_precision: "Were the retrieved chunks relevant? This grades the retriever, not the model.",
  context_recall: "Did retrieval find everything the reference answer needed?",
  factual_correctness: "Does the answer agree with the reference answer?",
};

/** Roughly how many LLM calls each metric costs per question. RAGAS does not
 *  make one call per metric: faithfulness extracts statements then verifies
 *  them, context precision grades every retrieved chunk separately, and factual
 *  correctness decomposes both the answer and the reference into claims.
 *  Used only to warn before a run — not a billing figure. */
export const METRIC_CALL_COST: Record<EvalMetric, number> = {
  faithfulness: 2,
  answer_relevancy: 1,
  context_precision: 3,
  context_recall: 1,
  factual_correctness: 4,
};

export interface TestSetSummary {
  name: string;
  source: string;
  size: number;
  has_references: boolean;
  metrics: EvalMetric[];
}

export interface EvalRow {
  question: string;
  ground_truth: string | null;
  answer: string | null;
  contexts: string[];
  route: string | null;
  grounded: boolean | null;
  latency_ms: number;
  scores: Partial<Record<EvalMetric, number>>;
  error: string | null;
}

export interface EvalRun {
  id: string;
  kind: "eval" | "generate";
  status: "running" | "completed" | "failed" | "cancelled";
  started_at: string;
  completed_at: string | null;
  testset: string | null;
  testset_source: string | null;
  testset_size: number;
  has_references: boolean;
  metrics: EvalMetric[];
  llm_provider: string | null;
  llm_model: string | null;
  embedding_provider: string | null;
  embedding_model: string | null;
  /** Which model produced the answers, when a separate judge graded them. */
  answered_by: string | null;
  /** Set when the judge model is too weak to be trusted; shown as a banner. */
  judge_warning: string | null;
  scores: Partial<Record<EvalMetric, number | null>>;
  error: string | null;
}

export interface EvalRunDetail extends EvalRun {
  rows: EvalRow[];
}

export interface ChatResponse {
  answer: string;
  route: string;
  sources: SourceDocument[];
  /** False when the answer came from the model's general knowledge because
   *  retrieval found nothing relevant in the user's documents. */
  grounded?: boolean;
}

export interface SearchResult {
  title: string;
  href: string;
  body: string;
}

export interface ConfigPayload {
  routing_method: string;
  reranker_provider: string;
  llm_provider?: string;
  llm_model?: string;
  embedding_provider?: string;
  embedding_model?: string;
  /** Separate judge for evaluation. "" clears it back to the chat model. */
  eval_llm_provider?: string;
  eval_llm_model?: string;
  openai_key?: string | null;
  anthropic_key?: string | null;
  google_key?: string | null;
  xai_key?: string | null;
  cohere_key?: string | null;
}

/** FastAPI returns errors as `{ detail: string }`; surface that text, not a status code. */
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** 401 means "sign in again", not "something broke" — callers redirect on this. */
  get isUnauthorized() {
    return this.status === 401;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  // Every call carries the session token. The backend derives the workspace from
  // its verified `sub` claim, so identity is never something we send in a body.
  const token = await accessToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(
      `Cannot reach the API at ${API_BASE || window.location.origin}. Is the server running?`,
      0,
    );
  }
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    throw new ApiError(data?.detail ?? `Request failed (${res.status})`, res.status);
  }
  return data as T;
}

/** Public: tells the client whether a sign-in screen is needed at all. */
export interface AuthConfig {
  auth_enabled: boolean;
  supabase_url: string | null;
}

export async function fetchAuthConfig(): Promise<AuthConfig> {
  const res = await fetch(`${API_BASE}/api/auth/config`);
  if (!res.ok) return { auth_enabled: false, supabase_url: null };
  return res.json();
}

export const api = {
  status: () => request<StatusResponse>("/api/status"),

  providers: () => request<ProviderCatalog>("/api/providers"),

  chat: (message: string, history: { role: string; content: string }[]) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    }),

  /**
   * Streaming variant of chat — fires onToken for each chunk, then onDone
   * with the final metadata once the server closes the SSE stream.
   * Uses the bearer token from Supabase (same as the regular chat call).
   */
  chatStream: async (
    message: string,
    history: { role: string; content: string }[],
    onToken: (token: string) => void,
    onDone: (meta: { route: string; sources: SourceDocument[]; grounded: boolean }) => void,
    onError: (msg: string) => void,
    signal?: AbortSignal,
  ): Promise<void> => {
    const token = await accessToken();
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) headers["Authorization"] = `Bearer ${token}`;

    let res: Response;
    try {
      res = await fetch(`${API_BASE}/api/chat/stream`, {
        method: "POST",
        headers,
        body: JSON.stringify({ message, history }),
        signal,
      });
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      throw new ApiError(
        `Cannot reach the API at ${API_BASE || window.location.origin}. Is the server running?`,
        0,
      );
    }

    if (!res.ok) {
      const text = await res.text();
      const data = text ? JSON.parse(text) : null;
      throw new ApiError(data?.detail ?? `Request failed (${res.status})`, res.status);
    }

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";
        for (const part of parts) {
          if (!part.startsWith("data: ")) continue;
          let payload: Record<string, unknown>;
          try {
            payload = JSON.parse(part.slice(6));
          } catch {
            continue;
          }
          if (payload.error) {
            onError(String(payload.error));
            return;
          }
          if (payload.done) {
            onDone({
              route: String(payload.route ?? "unknown"),
              sources: (payload.sources as SourceDocument[]) ?? [],
              grounded: payload.grounded !== false,
            });
          } else if (payload.token != null) {
            onToken(String(payload.token));
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  },

  config: (payload: ConfigPayload) =>
    request<{ status: string; message: string; embedding_changed: boolean }>("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  upload: (files: FileList | File[]) => {
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    return request<{ status: string; uploaded_files: string[] }>("/api/upload", {
      method: "POST",
      body: form,
    });
  },

  ingest: (raptor: boolean) =>
    request<{ status: string; build_id: string }>(
      `/api/ingest?raptor=${raptor}`,
      { method: "POST" },
    ),

  listBuilds: () => request<{ builds: BuildMeta[] }>("/api/builds"),
  
  cancelBuild: (build_id: string) => 
    request<{ status: string }>(`/api/builds/${build_id}/cancel`, { method: "POST" }),

  deleteDocument: (filename: string) =>
    request<{ status: string; deleted: string }>(
      `/api/documents/${encodeURIComponent(filename)}`,
      { method: "DELETE" },
    ),

  search: (q: string) =>
    request<{ results: SearchResult[] }>(`/api/search?q=${encodeURIComponent(q)}`),

  // ── evaluation ───────────────────────────────────────────────────────────

  listTestsets: () => request<TestSetSummary[]>("/api/eval/testsets"),

  /** Save a test set from pasted/uploaded JSON or CSV text. */
  createTestset: (name: string, content: string, source = "uploaded") =>
    request<TestSetSummary>("/api/eval/testsets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, content, source }),
    }),

  /** Save a set of bare questions — used for questions lifted from chat history,
   *  which have no reference answers and so score on fewer metrics. */
  createTestsetFromQuestions: (name: string, questions: string[], source = "chat") =>
    request<TestSetSummary>("/api/eval/testsets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, questions, source }),
    }),

  deleteTestset: (name: string) =>
    request<{ status: string; deleted: string }>(
      `/api/eval/testsets/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    ),

  generateTestset: (name: string, size: number) =>
    request<{ status: string; run_id: string }>("/api/eval/testsets/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, size }),
    }),

  /** Omit `metrics` to score everything the test set supports. Naming a subset
   *  is the main cost lever: each metric is several judge calls per question. */
  runEval: (testset: string, metrics?: EvalMetric[]) =>
    request<{ status: string; run_id: string }>("/api/eval/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ testset, metrics }),
    }),

  listEvalRuns: () => request<EvalRun[]>("/api/eval/runs"),

  getEvalRun: (runId: string) => request<EvalRunDetail>(`/api/eval/runs/${runId}`),

  cancelEvalRun: (runId: string) =>
    request<{ status: string }>(`/api/eval/runs/${runId}/cancel`, { method: "POST" }),
};
