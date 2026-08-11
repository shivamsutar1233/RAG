"use client";

import { useMemo } from "react";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { METRIC_LABELS, type EvalMetric, type EvalRun } from "@/lib/api";

/**
 * Metric scores across evaluation runs.
 *
 * One y-axis, because every RAGAS metric is already a 0–1 fraction — a second
 * scale would let two lines cross without meaning anything.
 *
 * Colours come from --chart-1..5 in fixed metric order, so a metric keeps its
 * colour whether or not the reference-based ones are in the run. Those tokens
 * are validated for colourblind separation and for both surfaces; three of the
 * light-mode steps sit under 3:1, which is why the legend is always drawn and
 * the same numbers appear as text in the scorecard tiles and the question table.
 */

interface Props {
  /** Completed runs, newest first (as the API returns them). */
  runs: EvalRun[];
  metrics: EvalMetric[];
}

type Point = { label: string; judge: string } & Partial<Record<EvalMetric, number>>;

export function EvalTrendChart({ runs, metrics }: Props) {
  const { data, series } = useMemo(() => {
    const ordered = [...runs].reverse(); // oldest first: time reads left to right
    const points: Point[] = ordered.map((run, i) => {
      const point: Point = {
        label: `#${i + 1}`,
        judge: `${run.llm_provider ?? "?"}/${run.llm_model ?? "?"}`,
      };
      for (const metric of metrics) {
        const value = run.scores?.[metric];
        if (typeof value === "number") point[metric] = value;
      }
      return point;
    });
    // Only plot metrics that actually have data, so a run without references
    // does not draw four flat empty lines.
    const present = metrics.filter((m) => points.some((p) => typeof p[m] === "number"));
    return { data: points, series: present };
  }, [runs, metrics]);

  if (data.length < 2 || series.length === 0) return null;

  return (
    <div>
      <div style={{ width: "100%", height: 260 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
            {/* Recessive: horizontal rules only, no vertical clutter. */}
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="label"
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
            />
            <YAxis
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tickLine={false}
              axisLine={false}
              tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
            />
            <Tooltip
              cursor={{ stroke: "var(--border)", strokeWidth: 1 }}
              content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null;
                const judge = (payload[0]?.payload as Point | undefined)?.judge;
                return (
                  <div className="bg-popover rounded-lg border p-2.5 text-xs shadow-md">
                    <p className="mb-1.5 font-medium">Run {label}</p>
                    {payload.map((entry) => (
                      <p key={String(entry.dataKey)} className="flex items-center gap-2">
                        <span
                          aria-hidden
                          className="size-2 shrink-0 rounded-full"
                          style={{ backgroundColor: entry.color }}
                        />
                        {/* Label and value wear text tokens; only the dot carries identity. */}
                        <span className="text-muted-foreground">
                          {METRIC_LABELS[entry.dataKey as EvalMetric]}
                        </span>
                        <span className="tabular ml-auto font-medium">
                          {typeof entry.value === "number" ? entry.value.toFixed(2) : "—"}
                        </span>
                      </p>
                    ))}
                    {judge && (
                      <p className="text-muted-foreground mt-1.5 border-t pt-1.5">
                        judge: {judge}
                      </p>
                    )}
                  </div>
                );
              }}
            />
            {series.map((metric) => (
              <Line
                key={metric}
                type="monotone"
                dataKey={metric}
                stroke={`var(--chart-${metrics.indexOf(metric) + 1})`}
                strokeWidth={2}
                dot={{ r: 3, strokeWidth: 0 }}
                activeDot={{ r: 5 }}
                connectNulls
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Always present for two or more series — identity is never colour alone. */}
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5">
        {series.map((metric) => (
          <span key={metric} className="text-muted-foreground flex items-center gap-1.5 text-xs">
            <span
              aria-hidden
              className="size-2.5 shrink-0 rounded-full"
              style={{ backgroundColor: `var(--chart-${metrics.indexOf(metric) + 1})` }}
            />
            {METRIC_LABELS[metric]}
          </span>
        ))}
      </div>
    </div>
  );
}
