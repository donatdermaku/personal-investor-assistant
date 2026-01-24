"use client";

import { MetricCard } from "@/components/ui/MetricCard";
import { definitionTooltip } from "@/lib/definitions";
import { useNexus } from "@/components/nexus/NexusProvider";
import { EmptyState } from "@/components/nexus/EmptyState";
import { SkeletonCard, SkeletonBlock } from "@/components/nexus/Skeleton";
import { SectionContext } from "@/components/nexus/SectionContext";
import { getMetricReasons, getMetricStatus } from "@/lib/coverageLogic";
import { LineChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function OverviewPage() {
    const { state, status, error, mode, setMode, openRunCreator, lastRunCreated, clearRunCreated } = useNexus();

    if (status === "error") {
        return (
            <EmptyState
                title="Unable to load portfolio data"
                description={error || "Check your backend connection and try again."}
                primaryAction={{ label: "Retry in Live Mode", onClick: () => setMode("live") }}
                secondaryAction={{ label: "Switch to Demo Mode", onClick: () => setMode("demo") }}
            />
        );
    }

    if (status === "empty") {
        return (
            <EmptyState
                title="No portfolio runs yet"
                description="Run the pipeline to generate your first set of analytics, or switch to demo mode to preview Nexus."
                primaryAction={{ label: "Create Run", onClick: openRunCreator }}
                secondaryAction={{ label: "Switch to Demo Mode", onClick: () => setMode("demo") }}
            />
        );
    }

    if (status === "loading" || !state) {
        return (
            <div className="space-y-8">
                <div>
                    <SkeletonBlock className="h-8 w-40" />
                    <SkeletonBlock className="mt-2 h-4 w-56" />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <SkeletonCard />
                    <SkeletonCard />
                    <SkeletonCard />
                    <SkeletonCard />
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                    <SkeletonBlock className="h-5 w-32" />
                    <SkeletonBlock className="mt-4 h-64 w-full" />
                </div>
            </div>
        );
    }

    const { summary, equity_curve, definitions, manifest, coverage_summary } = state;
    const equitySeries = equity_curve as Array<{ date: string; value: number; benchmark?: number | null }>;
    const hasBenchmark = equitySeries.some((point) => point.benchmark != null);
    const metricStatus = (kpiKey: string) => getMetricStatus(kpiKey, coverage_summary ?? null);
    const metricReasons = (kpiKey: string) => getMetricReasons(kpiKey, coverage_summary ?? null);
    const metricCoverage = (kpiKey: string) => (metricStatus(kpiKey) === "insufficient" ? "insufficient" : undefined);
    const metricReasonCodes = (kpiKey: string) =>
        metricStatus(kpiKey) === "insufficient" ? metricReasons(kpiKey) : undefined;
    const asOf = summary.last_date || manifest.timestamp;

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold text-[#0F172A] mb-2">Overview</h2>
                <p className="text-gray-600">
                    {mode === "demo" ? "Demo portfolio performance summary" : "Portfolio performance summary"}
                </p>
            </div>
            {lastRunCreated && (
                <div className="rounded-lg border border-[#E3E7EE] bg-[#F2F6FF] px-4 py-3 text-sm text-[#1E40AF] flex items-center justify-between">
                    <div>
                        New run created · {lastRunCreated.run_id.slice(0, 8)} ·{" "}
                        {lastRunCreated.timestamp ? new Date(lastRunCreated.timestamp).toLocaleString() : "just now"}
                    </div>
                    <button
                        type="button"
                        onClick={clearRunCreated}
                        className="text-xs font-semibold"
                    >
                        Dismiss
                    </button>
                </div>
            )}

            <SectionContext
                title="Overview Context"
                items={[
                    {
                        label: "What it measures",
                        text: "High-level portfolio results: total value, time-weighted and money-weighted returns, and max drawdown.",
                    },
                    {
                        label: "Why it matters",
                        text: "Sets the baseline for performance and risk conversations across the rest of the app.",
                    },
                    {
                        label: "When it misleads",
                        text: "Short histories or sparse pricing can understate drawdowns and distort return rates.",
                    },
                    {
                        label: "Assumptions",
                        text: "Returns assume end-of-day prices and rely on definitions for TWR, MWR, and drawdown.",
                    },
                ]}
            />

            {/* KPI Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <MetricCard
                    label="Strategy Return (TWR)"
                    value={summary.twr !== null ? `${(summary.twr * 100).toFixed(2)}%` : "--"}
                    tooltip={definitionTooltip(definitions, "twr")}
                    coverageStatus={metricCoverage("twr")}
                    asOf={asOf}
                    reasonCodes={metricReasonCodes("twr")}
                />
                <MetricCard
                    label="Personal Return (MWR)"
                    value={summary.mwr !== null ? `${(summary.mwr * 100).toFixed(2)}%` : "--"}
                    tooltip={definitionTooltip(definitions, "mwr")}
                    coverageStatus={metricCoverage("mwr")}
                    asOf={asOf}
                    reasonCodes={metricReasonCodes("mwr")}
                />
                <MetricCard
                    label="Portfolio Value"
                    value={summary.final_value !== null ? `$${summary.final_value.toLocaleString()}` : "--"}
                    subtext={summary.last_date || undefined}
                    coverageStatus={metricCoverage("portfolio_value")}
                    asOf={asOf}
                    reasonCodes={metricReasonCodes("portfolio_value")}
                />
                <MetricCard
                    label="Max Drawdown"
                    value={summary.max_drawdown !== null ? `${(summary.max_drawdown * 100).toFixed(2)}%` : "--"}
                    tooltip={definitionTooltip(definitions, "max_drawdown")}
                    coverageStatus={metricCoverage("max_drawdown")}
                    asOf={asOf}
                    reasonCodes={metricReasonCodes("max_drawdown")}
                />
            </div>

            {/* Equity Curve */}
            <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-lg font-semibold text-[#0F172A] mb-4">Equity Curve</h3>
                {equity_curve.length === 0 ? (
                    <div className="text-sm text-gray-500">No equity curve data available.</div>
                ) : (
                    <ResponsiveContainer width="100%" height={400}>
                        <LineChart data={equitySeries}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#E3E7EE" />
                            <XAxis
                                dataKey="date"
                                stroke="#64748B"
                                style={{ fontSize: '12px' }}
                            />
                            <YAxis
                                stroke="#64748B"
                                style={{ fontSize: '12px' }}
                                tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'white',
                                    border: '1px solid #E3E7EE',
                                    borderRadius: '0.5rem'
                                }}
                                formatter={(value) => value != null ? [`$${Number(value).toLocaleString()}`, 'Value'] : ['--', 'Value']}
                            />
                            <Legend />
                            <Area
                                type="monotone"
                                dataKey="value"
                                stroke="none"
                                fill="rgba(37, 99, 235, 0.08)"
                                name="Portfolio Value"
                            />
                            {hasBenchmark && (
                                <Line
                                    type="monotone"
                                    dataKey="benchmark"
                                    stroke="#94A3B8"
                                    strokeWidth={2}
                                    strokeDasharray="4 4"
                                    dot={false}
                                    name="Benchmark"
                                />
                            )}
                            <Line
                                type="monotone"
                                dataKey="value"
                                stroke="#2563EB"
                                strokeWidth={2}
                                dot={false}
                                name="Portfolio Value"
                            />
                        </LineChart>
                    </ResponsiveContainer>
                )}
            </div>

            {/* Errors */}
            {summary.errors && summary.errors.length > 0 && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <h4 className="font-semibold text-red-900 mb-2">Data issues detected</h4>
                    <ul className="list-disc list-inside text-red-700 text-sm space-y-1">
                        {summary.errors.map((err, idx) => (
                            <li key={idx}>{err}</li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}
