"use client";

import { useEffect, useMemo, useState } from "react";

type Experiment = {
  id: string;
  name: string;
  model?: string | null;
  evaluator?: string | null;
  dataset_version?: string | null;
  status: string;
  created_at: string;
  metrics: Record<string, number | string | boolean>;
  metadata: Record<string, unknown>;
};

const fallbackExperiments: Experiment[] = [
  {
    id: "exp-101",
    name: "support-v2",
    model: "gpt-4o",
    evaluator: "exact_match",
    dataset_version: "2026-10",
    status: "completed",
    created_at: "2026-10-05T16:00:00Z",
    metrics: { exact_match: 0.92, retrieval_precision: 0.88 },
    metadata: { quality_gates: { passed: true } },
  },
  {
    id: "exp-102",
    name: "customer-service",
    model: "gpt-4.1-mini",
    evaluator: "judged_faithfulness",
    dataset_version: "2026-09",
    status: "completed",
    created_at: "2026-10-05T12:20:00Z",
    metrics: { judged_faithfulness: 0.91, answer_relevance: 0.86 },
    metadata: { quality_gates: { passed: true } },
  },
  {
    id: "exp-103",
    name: "risk-scan",
    model: "claude-3.5",
    evaluator: "quality_gate",
    dataset_version: "2026-09",
    status: "failed",
    created_at: "2026-10-04T21:15:00Z",
    metrics: { exact_match: 0.71, confidence: 0.64 },
    metadata: { quality_gates: { passed: false } },
  },
];

function formatDate(value: string) {
  return new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export default function Home() {
  const [experiments, setExperiments] = useState<Experiment[]>(fallbackExperiments);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadExperiments = async () => {
      try {
        const response = await fetch("http://127.0.0.1:8000/api/v1/experiments?limit=10");
        if (!response.ok) {
          throw new Error("Experiments API unavailable");
        }
        const payload = await response.json();
        if (Array.isArray(payload.items) && payload.items.length > 0) {
          setExperiments(payload.items as Experiment[]);
        }
      } catch {
        setExperiments(fallbackExperiments);
      } finally {
        setLoading(false);
      }
    };

    loadExperiments();
  }, []);

  const summary = useMemo(() => {
    const numericValues = experiments
      .flatMap((experiment) => Object.values(experiment.metrics))
      .filter((value): value is number => typeof value === "number" && Number.isFinite(value));

    const averageScore = numericValues.length
      ? numericValues.reduce((sum, value) => sum + value, 0) / numericValues.length
      : 0;

    const passedRuns = experiments.filter((experiment) => {
      const gate = experiment.metadata?.quality_gates as { passed?: boolean } | undefined;
      return experiment.status === "completed" || gate?.passed === true;
    }).length;

    return {
      total: experiments.length,
      passRate: experiments.length ? (passedRuns / experiments.length) * 100 : 0,
      avgScore: averageScore * 100,
    };
  }, [experiments]);

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-6 py-10">
        <header className="mb-8 flex items-center justify-between gap-4 border-b border-slate-800 pb-6">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-cyan-400">EvalOm</p>
            <h1 className="mt-2 text-3xl font-semibold">Evaluation trust dashboard</h1>
          </div>
          <div className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1 text-sm text-cyan-300">
            {loading ? "Refreshing…" : "Live data"}
          </div>
        </header>

        <section className="mb-8 grid gap-4 md:grid-cols-3">
          <StatCard label="Total runs" value={String(summary.total)} accent="cyan" />
          <StatCard label="Pass rate" value={`${summary.passRate.toFixed(0)}%`} accent="green" />
          <StatCard label="Average score" value={`${summary.avgScore.toFixed(1)}%`} accent="violet" />
        </section>

        <section className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/80 shadow-2xl shadow-slate-950/30">
          <div className="border-b border-slate-800 px-6 py-4">
            <h2 className="text-lg font-medium">Recent experiments</h2>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full text-left">
              <thead className="bg-slate-950/70 text-sm uppercase tracking-[0.12em] text-slate-400">
                <tr>
                  <th className="px-6 py-3">Run</th>
                  <th className="px-6 py-3">Model</th>
                  <th className="px-6 py-3">Status</th>
                  <th className="px-6 py-3">Score</th>
                  <th className="px-6 py-3">Created</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((experiment) => {
                  const metricValues = Object.values(experiment.metrics).filter(
                    (value): value is number => typeof value === "number" && Number.isFinite(value),
                  );
                  const score = metricValues.length
                    ? (metricValues.reduce((sum, value) => sum + value, 0) / metricValues.length) * 100
                    : 0;

                  return (
                    <tr key={experiment.id} className="border-t border-slate-800 text-sm text-slate-200">
                      <td className="px-6 py-4">
                        <div className="font-medium text-white">{experiment.name}</div>
                        <div className="text-xs text-slate-400">{experiment.id}</div>
                      </td>
                      <td className="px-6 py-4">{experiment.model ?? "—"}</td>
                      <td className="px-6 py-4">
                        <span
                          className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                            experiment.status === "failed"
                              ? "bg-red-500/10 text-red-300"
                              : "bg-emerald-500/10 text-emerald-300"
                          }`}
                        >
                          {experiment.status}
                        </span>
                      </td>
                      <td className="px-6 py-4">{score.toFixed(1)}%</td>
                      <td className="px-6 py-4 text-slate-400">{formatDate(experiment.created_at)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </main>
  );
}

function StatCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent: "cyan" | "green" | "violet";
}) {
  const accentMap = {
    cyan: "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
    green: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
    violet: "border-violet-500/30 bg-violet-500/10 text-violet-300",
  };

  return (
    <div className={`rounded-2xl border p-5 ${accentMap[accent]}`}>
      <div className="text-sm uppercase tracking-[0.16em] text-slate-300">{label}</div>
      <div className="mt-3 text-3xl font-semibold text-white">{value}</div>
    </div>
  );
}
