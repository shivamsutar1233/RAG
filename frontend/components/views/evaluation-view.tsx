"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle, ChevronDown, FlaskConical, Loader2, MessageSquare, Play,
  Sparkles, Trash2, Upload, X,
} from "lucide-react";
import { toast } from "sonner";
import {
  API_BASE, ApiError, METRIC_CALL_COST, METRIC_HELP, METRIC_LABELS, api,
  type EvalMetric, type EvalRun, type EvalRunDetail, type StatusResponse, type TestSetSummary,
} from "@/lib/api";
import { accessToken } from "@/lib/supabase";
import type { Session } from "@/lib/sessions";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { EvalTrendChart } from "@/components/eval-trend-chart";
import { cn } from "@/lib/utils";

/** Fixed order, matching the chart's colour slots. A metric always keeps its
 *  colour whether or not the reference-based ones are present. */
const METRIC_ORDER: EvalMetric[] = [
  "faithfulness",
  "answer_relevancy",
  "context_precision",
  "context_recall",
  "factual_correctness",
];

/** Below this a metric is flagged in the table. Not a pass/fail verdict — it is
 *  the point at which a row is worth reading yourself. */
const REVIEW_BELOW = 0.6;

type LogLine = { text: string; tone: "info" | "ok" | "err" | "warn" };

function fmt(value: number | null | undefined) {
  return value === null || value === undefined ? "—" : value.toFixed(2);
}

function timeAgo(date: string) {
  const diff = (Date.now() - new Date(date).getTime()) / 1000;
  if (!Number.isFinite(diff)) return "";
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function ScoreTile({ metric, value }: { metric: EvalMetric; value: number | null | undefined }) {
  const slot = METRIC_ORDER.indexOf(metric) + 1;
  return (
    <Card className="gap-0 py-5">
      <CardContent className="px-5">
        <div className="text-muted-foreground mb-2.5 flex items-center gap-2">
          {/* The swatch ties the tile to its line in the trend chart. Identity is
              never colour alone — the label sits right beside it. */}
          <span
            aria-hidden
            className="size-2.5 shrink-0 rounded-full"
            style={{ backgroundColor: `var(--chart-${slot})` }}
          />
          <span className="text-[11.5px] font-semibold tracking-wider uppercase">
            {METRIC_LABELS[metric]}
          </span>
        </div>
        <p className="tabular text-2xl font-semibold tracking-tight">{fmt(value)}</p>
        <p className="text-muted-foreground mt-1.5 text-[11.5px] leading-snug">
          {METRIC_HELP[metric]}
        </p>
      </CardContent>
    </Card>
  );
}

function StatusPill({ status }: { status: EvalRun["status"] }) {
  const tone =
    status === "completed"
      ? "text-success"
      : status === "running"
        ? "text-muted-foreground"
        : "text-destructive";
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-xs font-medium", tone)}>
      {status === "running" && <Loader2 className="size-3 animate-spin" />}
      {status}
    </span>
  );
}

interface Props {
  status: StatusResponse | null;
  sessions: Session[];
  active: boolean;
}

export function EvaluationView({ status, sessions, active }: Props) {
  const [testsets, setTestsets] = useState<TestSetSummary[]>([]);
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [selectedSet, setSelectedSet] = useState<string>("");
  const [detail, setDetail] = useState<EvalRunDetail | null>(null);
  const [openRow, setOpenRow] = useState<number | null>(null);
  const [log, setLog] = useState<LogLine[]>([]);
  const [busy, setBusy] = useState(false);
  const [genSize, setGenSize] = useState(5);
  const [genName, setGenName] = useState("");
  const [loaded, setLoaded] = useState(false);
  // null means "everything this set supports" — the backend's own default.
  const [chosenMetrics, setChosenMetrics] = useState<EvalMetric[] | null>(null);

  const fileRef = useRef<HTMLInputElement>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  const activeRun = runs.find((r) => r.status === "running");
  const hasIndex = status?.database_loaded ?? false;

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ block: "end" });
  }, [log]);

  const refresh = useCallback(async () => {
    try {
      const [sets, runList] = await Promise.all([api.listTestsets(), api.listEvalRuns()]);
      setTestsets(sets);
      setRuns(runList);
      setSelectedSet((current) => current || sets[0]?.name || "");
      setLoaded(true);
    } catch (err) {
      if (err instanceof ApiError && err.isUnauthorized) return;
      setLoaded(true);
    }
  }, []);

  // All views stay mounted and are hidden with CSS, so an ungated interval would
  // keep polling the API from a tab nobody is looking at.
  useEffect(() => {
    if (!active) return;
    refresh();
    const interval = setInterval(refresh, activeRun ? 3000 : 15000);
    return () => clearInterval(interval);
  }, [active, refresh, activeRun ? activeRun.id : null]);

  // Show the newest finished evaluation without the user having to pick it.
  const latestEval = useMemo(
    () => runs.find((r) => r.kind === "eval" && r.status !== "running"),
    [runs],
  );

  // Fetch straight into place rather than clearing first — blanking the
  // scorecard between runs makes a finished run look like it failed.
  useEffect(() => {
    if (!latestEval) return;
    let stale = false;
    api
      .getEvalRun(latestEval.id)
      .then((full) => {
        if (!stale) setDetail(full);
      })
      .catch(() => {});
    return () => {
      stale = true;
    };
  }, [latestEval?.id]);

  // Tail the running job's log.
  useEffect(() => {
    if (!activeRun) return;
    const controller = new AbortController();

    (async () => {
      try {
        const token = await accessToken();
        const headers: Record<string, string> = {};
        if (token) headers["Authorization"] = `Bearer ${token}`;
        const res = await fetch(`${API_BASE}/api/eval/runs/${activeRun.id}/stream`, {
          headers,
          signal: controller.signal,
        });
        if (!res.body) return;
        setLog([]);

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";
          for (const frame of frames) {
            if (frame.startsWith("event: close")) return;
            if (!frame.startsWith("data: ")) continue;
            const payload = frame.slice(6).trim();
            if (!payload) continue;
            try {
              const data = JSON.parse(payload);
              if (!data.text) continue;
              const text = String(data.text).trimEnd();
              const lower = text.toLowerCase();
              const tone: LogLine["tone"] = lower.includes("[error]") || lower.includes("failed")
                ? "err"
                : lower.includes("[warning]")
                  ? "warn"
                  : lower.includes("[done]")
                    ? "ok"
                    : "info";
              setLog((lines) => [...lines, { text, tone }]);
            } catch {}
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          setLog((lines) => [...lines, { text: "[stream disconnected]", tone: "err" }]);
        }
      }
    })();

    return () => controller.abort();
  }, [activeRun?.id]);

  async function guard(action: () => Promise<void>) {
    setBusy(true);
    try {
      await action();
      await refresh();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  const startRun = () =>
    guard(async () => {
      if (!selectedSet) throw new ApiError("Pick a test set first.", 400);
      setLog([{ text: "Starting evaluation...", tone: "info" }]);
      await api.runEval(selectedSet, chosenMetrics ?? undefined);
    });

  const startGeneration = () =>
    guard(async () => {
      const name = (genName.trim() || `generated-${new Date().toISOString().slice(0, 10)}`).slice(0, 64);
      setLog([{ text: `Generating '${name}'...`, tone: "info" }]);
      await api.generateTestset(name, genSize);
      setGenName("");
    });

  const importFile = (file: File) =>
    guard(async () => {
      const text = await file.text();
      const name = file.name.replace(/\.(json|csv|txt)$/i, "").slice(0, 64);
      const created = await api.createTestset(name, text);
      setSelectedSet(created.name);
      toast.success(`Imported ${created.size} question(s).`);
    });

  const importFromChat = () =>
    guard(async () => {
      const questions = Array.from(
        new Set(
          sessions
            .flatMap((s) => s.messages)
            .filter((m) => m.role === "user" && m.content.trim().length > 8)
            .map((m) => m.content.trim()),
        ),
      ).slice(0, 50);
      if (!questions.length) throw new ApiError("No chat questions to import yet.", 400);
      const name = `chat-${new Date().toISOString().slice(0, 10)}`;
      const created = await api.createTestsetFromQuestions(name, questions);
      setSelectedSet(created.name);
      toast.success(
        `Imported ${created.size} question(s). No reference answers, so these score on ${created.metrics.length} metrics.`,
      );
    });

  const removeTestset = (name: string) =>
    guard(async () => {
      await api.deleteTestset(name);
      setSelectedSet((current) => (current === name ? "" : current));
    });

  const cancelRun = () =>
    guard(async () => {
      if (activeRun) await api.cancelEvalRun(activeRun.id);
    });

  const chosen = testsets.find((t) => t.name === selectedSet);
  const shownMetrics = detail?.metrics?.length ? detail.metrics : METRIC_ORDER;

  // Metrics actually available for the selected set, and the subset that will run.
  const availableMetrics = chosen?.metrics ?? METRIC_ORDER;
  const runMetrics = chosenMetrics
    ? availableMetrics.filter((m) => chosenMetrics.includes(m))
    : availableMetrics;
  const effectiveMetrics = runMetrics.length ? runMetrics : availableMetrics;

  // RAGAS issues several judge calls per metric per question, plus the RAG query
  // itself. Surfaced before the run because on a rate-limited tier this number,
  // not the question count, is what decides whether the run finishes.
  const estimatedCalls =
    (chosen?.size ?? 0) *
    (effectiveMetrics.reduce((sum, m) => sum + METRIC_CALL_COST[m], 0) + 6);

  const toggleMetric = (metric: EvalMetric) =>
    setChosenMetrics((current) => {
      const base = current ?? availableMetrics;
      const next = base.includes(metric)
        ? base.filter((m) => m !== metric)
        : [...base, metric];
      // Deselecting everything means "all" again rather than an unrunnable run.
      return next.length ? next : null;
    });
  const evalRuns = useMemo(
    () => runs.filter((r) => r.kind === "eval" && r.status === "completed"),
    [runs],
  );

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-6xl space-y-6 p-6 pb-16">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Evaluation</h1>
          <p className="text-muted-foreground mt-1 text-sm">
            Score this workspace&apos;s retrieval and answers with RAGAS, using the models
            you have configured. Nothing leaves your deployment.
          </p>
        </div>

        {!hasIndex && (
          <Card className="border-warning/40">
            <CardContent className="flex items-start gap-3 py-4">
              <AlertTriangle className="text-warning mt-0.5 size-4 shrink-0" />
              <p className="text-sm">
                No index yet. Upload documents and build the index on the Documents tab
                before running an evaluation.
              </p>
            </CardContent>
          </Card>
        )}

        {detail?.judge_warning && (
          <Card className="border-warning/40">
            <CardContent className="flex items-start gap-3 py-4">
              <AlertTriangle className="text-warning mt-0.5 size-4 shrink-0" />
              <div>
                <p className="text-sm font-medium">Judge model warning</p>
                <p className="text-muted-foreground mt-0.5 text-sm">{detail.judge_warning}</p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* ── Test sets ─────────────────────────────────────────────────── */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Test sets</CardTitle>
            <CardDescription>
              A test set is the list of questions to score. Sets with reference answers
              are graded on all five metrics; sets without them on three.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="flex flex-wrap items-end gap-3">
              <div className="grow space-y-1.5" style={{ minWidth: "12rem" }}>
                <Label htmlFor="gen-name">Generate from my documents</Label>
                <Input
                  id="gen-name"
                  placeholder="name (optional)"
                  value={genName}
                  onChange={(e) => setGenName(e.target.value)}
                />
              </div>
              <div className="w-24 space-y-1.5">
                <Label htmlFor="gen-size">Questions</Label>
                <Input
                  id="gen-size"
                  type="number"
                  min={1}
                  max={50}
                  value={genSize}
                  onChange={(e) => setGenSize(Math.max(1, Math.min(50, +e.target.value || 1)))}
                />
              </div>
              <Button onClick={startGeneration} disabled={busy || !!activeRun || !hasIndex}>
                <Sparkles className="size-4" /> Generate
              </Button>
              <Button variant="outline" onClick={() => fileRef.current?.click()} disabled={busy}>
                <Upload className="size-4" /> Import JSON/CSV
              </Button>
              <Button variant="outline" onClick={importFromChat} disabled={busy}>
                <MessageSquare className="size-4" /> From chat history
              </Button>
              <input
                ref={fileRef}
                type="file"
                accept=".json,.csv,.txt"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) importFile(file);
                  e.target.value = "";
                }}
              />
            </div>

            {!loaded ? (
              <Skeleton className="h-16 w-full" />
            ) : testsets.length === 0 ? (
              <p className="text-muted-foreground text-sm">
                No test sets yet. Generate one from your documents to get started.
              </p>
            ) : (
              <div className="divide-border divide-y rounded-lg border">
                {testsets.map((set) => (
                  <div key={set.name} className="flex items-center gap-3 px-4 py-2.5">
                    <div className="min-w-0 grow">
                      <p className="truncate text-sm font-medium">{set.name}</p>
                      <p className="text-muted-foreground text-xs">
                        {set.size} question{set.size === 1 ? "" : "s"} · {set.source} ·{" "}
                        {set.has_references ? "with references" : "no references"} ·{" "}
                        {set.metrics.length} metrics
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      aria-label={`Delete ${set.name}`}
                      disabled={busy}
                      onClick={() => removeTestset(set.name)}
                    >
                      <Trash2 className="size-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* ── Run ───────────────────────────────────────────────────────── */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Run an evaluation</CardTitle>
            <CardDescription>
              Every question is asked against your real pipeline with the web fallback
              off, then the answer and its retrieved context are scored.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-end gap-3">
              <div className="grow space-y-1.5" style={{ minWidth: "14rem" }}>
                <Label>Test set</Label>
                <Select value={selectedSet} onValueChange={(v) => setSelectedSet(v ?? "")}>
                  <SelectTrigger>
                    <SelectValue placeholder="Pick a test set" />
                  </SelectTrigger>
                  <SelectContent>
                    {testsets.map((set) => (
                      <SelectItem key={set.name} value={set.name}>
                        {set.name} ({set.size})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {activeRun ? (
                <Button variant="destructive" onClick={cancelRun} disabled={busy}>
                  <X className="size-4" /> Cancel run
                </Button>
              ) : (
                <Button onClick={startRun} disabled={busy || !selectedSet || !hasIndex}>
                  <Play className="size-4" /> Run evaluation
                </Button>
              )}
            </div>

            {chosen && (
              <div className="space-y-2">
                <Label>Metrics</Label>
                <div className="flex flex-wrap gap-2">
                  {METRIC_ORDER.map((metric) => {
                    const supported = availableMetrics.includes(metric);
                    const on = supported && effectiveMetrics.includes(metric);
                    return (
                      <button
                        key={metric}
                        type="button"
                        disabled={!supported || !!activeRun}
                        onClick={() => toggleMetric(metric)}
                        title={
                          supported
                            ? METRIC_HELP[metric]
                            : "Needs reference answers in the test set."
                        }
                        className={cn(
                          "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                          on
                            ? "border-primary bg-primary/10 text-foreground"
                            : "text-muted-foreground",
                          !supported && "cursor-not-allowed opacity-40",
                        )}
                      >
                        <span
                          aria-hidden
                          className="mr-1.5 inline-block size-2 rounded-full align-middle"
                          style={{
                            backgroundColor: on
                              ? `var(--chart-${METRIC_ORDER.indexOf(metric) + 1})`
                              : "var(--muted-foreground)",
                          }}
                        />
                        {METRIC_LABELS[metric]}
                        <span className="text-muted-foreground ml-1.5">
                          ~{METRIC_CALL_COST[metric]}
                        </span>
                      </button>
                    );
                  })}
                </div>
                <p className="text-muted-foreground text-xs">
                  {chosen.has_references
                    ? "Numbers are roughly how many LLM calls each metric costs per question."
                    : "This set has no reference answers, so context recall and factual correctness cannot be computed."}{" "}
                  Estimated <span className="tabular font-medium">~{estimatedCalls}</span>{" "}
                  model calls for this run — check that against your provider&apos;s rate
                  limit before starting.
                </p>
              </div>
            )}

            {(activeRun || log.length > 0) && (
              <div className="bg-muted/40 max-h-64 overflow-auto rounded-lg border p-3 font-mono text-xs">
                {log.map((line, i) => (
                  <p
                    key={i}
                    className={cn(
                      "whitespace-pre-wrap",
                      line.tone === "err" && "text-destructive",
                      line.tone === "warn" && "text-warning",
                      line.tone === "ok" && "text-success",
                      line.tone === "info" && "text-muted-foreground",
                    )}
                  >
                    {line.text}
                  </p>
                ))}
                <div ref={logEndRef} />
              </div>
            )}
          </CardContent>
        </Card>

        {/* ── Scorecard ─────────────────────────────────────────────────── */}
        {detail && detail.kind === "eval" && (
          <>
            <div>
              <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
                <h2 className="text-lg font-semibold tracking-tight">Latest scorecard</h2>
                <p className="text-muted-foreground text-xs">
                  {detail.testset} · {detail.testset_size} questions ·{" "}
                  {detail.answered_by && detail.answered_by !== `${detail.llm_provider}/${detail.llm_model}`
                    ? `answered by ${detail.answered_by}, judged by ${detail.llm_provider}/${detail.llm_model}`
                    : `judged by ${detail.llm_provider}/${detail.llm_model}`}{" "}
                  · {timeAgo(detail.started_at)}
                </p>
              </div>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {shownMetrics.map((metric) => (
                  <ScoreTile key={metric} metric={metric} value={detail.scores[metric]} />
                ))}
              </div>
            </div>

            {evalRuns.length >= 2 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Trend</CardTitle>
                  <CardDescription>
                    Scores across completed runs, oldest first. A jump here only means
                    progress if the judge model stayed the same.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <EvalTrendChart runs={evalRuns} metrics={METRIC_ORDER} />
                </CardContent>
              </Card>
            )}

            {/* Per-question detail. This is also the "relief" the palette needs:
                every score is present as text, never colour alone. */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Questions</CardTitle>
                <CardDescription>
                  Click a row to see the answer and the exact chunks it was given.
                </CardDescription>
              </CardHeader>
              <CardContent className="px-0">
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-[42%]">Question</TableHead>
                        <TableHead>Route</TableHead>
                        {shownMetrics.map((m) => (
                          <TableHead key={m} className="text-right whitespace-nowrap">
                            {METRIC_LABELS[m].split(" ")[0]}
                          </TableHead>
                        ))}
                        <TableHead className="text-right">Latency</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {detail.rows.map((row, i) => (
                        <>
                          <TableRow
                            key={i}
                            className="cursor-pointer"
                            onClick={() => setOpenRow(openRow === i ? null : i)}
                          >
                            <TableCell className="max-w-0">
                              <div className="flex items-center gap-1.5">
                                <ChevronDown
                                  className={cn(
                                    "text-muted-foreground size-3.5 shrink-0 transition-transform",
                                    openRow === i && "rotate-180",
                                  )}
                                />
                                <span className="truncate">{row.question}</span>
                              </div>
                            </TableCell>
                            <TableCell className="text-muted-foreground text-xs">
                              {row.error ? (
                                <span className="text-destructive">failed</span>
                              ) : (
                                row.route
                              )}
                            </TableCell>
                            {shownMetrics.map((m) => {
                              const value = row.scores?.[m];
                              const low = typeof value === "number" && value < REVIEW_BELOW;
                              return (
                                <TableCell
                                  key={m}
                                  className={cn(
                                    "tabular text-right",
                                    low && "text-destructive font-medium",
                                  )}
                                >
                                  {fmt(value)}
                                </TableCell>
                              );
                            })}
                            <TableCell className="tabular text-muted-foreground text-right text-xs">
                              {(row.latency_ms / 1000).toFixed(1)}s
                            </TableCell>
                          </TableRow>
                          {openRow === i && (
                            <TableRow key={`${i}-detail`} className="hover:bg-transparent">
                              <TableCell colSpan={shownMetrics.length + 3} className="bg-muted/30">
                                <div className="space-y-3 py-2 text-sm">
                                  {row.error && (
                                    <p className="text-destructive">{row.error}</p>
                                  )}
                                  <div>
                                    <p className="text-muted-foreground mb-1 text-xs font-semibold tracking-wider uppercase">
                                      Answer
                                    </p>
                                    <p className="whitespace-pre-wrap">{row.answer || "—"}</p>
                                  </div>
                                  {row.ground_truth && (
                                    <div>
                                      <p className="text-muted-foreground mb-1 text-xs font-semibold tracking-wider uppercase">
                                        Reference
                                      </p>
                                      <p className="whitespace-pre-wrap">{row.ground_truth}</p>
                                    </div>
                                  )}
                                  <div>
                                    <p className="text-muted-foreground mb-1 text-xs font-semibold tracking-wider uppercase">
                                      Retrieved context ({row.contexts.length})
                                    </p>
                                    <div className="space-y-2">
                                      {row.contexts.map((chunk, c) => (
                                        <p
                                          key={c}
                                          className="bg-background text-muted-foreground rounded border p-2 text-xs"
                                        >
                                          {chunk.slice(0, 600)}
                                          {chunk.length > 600 ? "…" : ""}
                                        </p>
                                      ))}
                                      {row.contexts.length === 0 && (
                                        <p className="text-muted-foreground text-xs">
                                          Nothing retrieved.
                                        </p>
                                      )}
                                    </div>
                                  </div>
                                </div>
                              </TableCell>
                            </TableRow>
                          )}
                        </>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </>
        )}

        {/* ── History ───────────────────────────────────────────────────── */}
        {runs.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Run history</CardTitle>
            </CardHeader>
            <CardContent className="px-0">
              <div className="divide-border divide-y">
                {runs.slice(0, 15).map((run) => (
                  <button
                    key={run.id}
                    type="button"
                    className="hover:bg-muted/50 flex w-full items-center gap-3 px-6 py-2.5 text-left"
                    onClick={() =>
                      run.kind === "eval" &&
                      api.getEvalRun(run.id).then(setDetail).catch(() => {})
                    }
                  >
                    {run.kind === "generate" ? (
                      <Sparkles className="text-muted-foreground size-4 shrink-0" />
                    ) : (
                      <FlaskConical className="text-muted-foreground size-4 shrink-0" />
                    )}
                    <div className="min-w-0 grow">
                      <p className="truncate text-sm">
                        {run.kind === "generate" ? "Generated" : "Evaluated"}{" "}
                        <span className="font-medium">{run.testset}</span>
                      </p>
                      <p className="text-muted-foreground text-xs">
                        {timeAgo(run.started_at)}
                        {run.error ? ` · ${run.error}` : ""}
                      </p>
                    </div>
                    {run.kind === "eval" && run.status === "completed" && (
                      <span className="tabular text-muted-foreground hidden text-xs sm:inline">
                        faithfulness {fmt(run.scores.faithfulness)}
                      </span>
                    )}
                    <StatusPill status={run.status} />
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </ScrollArea>
  );
}
