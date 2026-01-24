"use client";

import { MetricCard } from "@/components/ui/MetricCard";
import { definitionTooltip } from "@/lib/definitions";
import { useNexus } from "@/components/nexus/NexusProvider";
import { EmptyState } from "@/components/nexus/EmptyState";
import { SkeletonCard, SkeletonBlock } from "@/components/nexus/Skeleton";
import { SectionContext } from "@/components/nexus/SectionContext";
import { getMetricReasons, getMetricStatus } from "@/lib/coverageLogic";
import { LineChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts";

export default function PerformancePage() {
    const { state, status, error, mode, setMode, openRunCreator } = useNexus();

    if (status === "error") {
        return (
            <EmptyState
                title="Unable to load performance data"
                description={error || "Check your backend connection and try again."}
                primaryAction={{ label: "Retry in Live Mode", onClick: () => setMode("live") }}
                secondaryAction={{ label: "Switch to Demo Mode", onClick: () => setMode("demo") }}
            />
        );
    }

    if (status === "empty") {
        return (
            <EmptyState
                title="No performance history yet"
                description="Generate a run to see returns, drawdowns, and attribution data."
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
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <SkeletonCard />
                    <SkeletonCard />
                    <SkeletonCard />
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-6">
                    <SkeletonBlock className="h-5 w-32" />
                    <SkeletonBlock className="mt-4 h-56 w-full" />
                </div>
            </div>
        );
    }

    const {
        summary,
        performance,
        monthly_returns,
        definitions,
        attribution_summary,
        attribution_timeseries,
        benchmark_comparison,
        benchmark_timeseries,
        macro,
        macro_regimes,
        manifest,
        coverage_summary,
    } = state;
    const performanceSeries = performance as Array<{ date: string; value: number | null; benchmark?: number | null; drawdown?: number | null }>;
    const hasBenchmark = performanceSeries.some((point) => point.benchmark != null);
    const firstValue = performance[0]?.value ?? null;
    const lastValue = performance[performance.length - 1]?.value ?? null;
    const totalReturn =
        firstValue && lastValue ? (lastValue / firstValue) - 1 : null;
    const attributionRows = attribution_summary?.per_asset?.slice(0, 5) ?? [];
    const macroFlags = macro?.flags ?? macro_regimes ?? [];
    const macroStatus = macro?.status ?? (macroFlags.length > 0 ? "sufficient" : "unavailable");
    const latestMacro = macroFlags.length > 0 ? macroFlags[macroFlags.length - 1] : null;
    const hasAttribution = attribution_summary && typeof attribution_summary.allocation === "number";
    const hasBenchmarkComparison = benchmark_comparison && typeof benchmark_comparison.tracking_error === "number";
    const metricStatus = (kpiKey: string) => getMetricStatus(kpiKey, coverage_summary ?? null);
    const metricReasons = (kpiKey: string) => getMetricReasons(kpiKey, coverage_summary ?? null);
    const metricCoverage = (kpiKey: string) => (metricStatus(kpiKey) === "insufficient" ? "insufficient" : undefined);
    const metricReasonCodes = (kpiKey: string) =>
        metricStatus(kpiKey) === "insufficient" ? metricReasons(kpiKey) : undefined;
    const asOf = summary.last_date || manifest.timestamp;

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold text-[#0F172A] mb-2">Performance</h2>
                <p className="text-gray-600">
                    {mode === "demo" ? "Demo returns analysis" : "Detailed returns analysis"}
                </p>
            </div>

            <SectionContext
                title="Performance Context"
                items={[
                    {
                        label: "What it measures",
                        text: "Portfolio returns over time, including time-weighted and money-weighted views.",
                    },
                    {
                        label: "Why it matters",
                        text: "Separates market movement from cash-flow timing so results are comparable over time.",
                    },
                    {
                        label: "When it misleads",
                        text: "Sparse price history or short windows can understate drawdowns and overstate volatility.",
                    },
                    {
                        label: "Assumptions",
                        text: "Returns assume end-of-day prices and follow the TWR/MWR definitions.",
                    },
                ]}
            />

            {summary.errors && summary.errors.length > 0 && (
                <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                    Some inputs are missing or incomplete for this run. Metrics relying on those inputs may be limited.
                </div>
            )}

            {/* Returns Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <MetricCard
                    label="TWR (Strategy)"
                    value={summary.twr !== null ? `${(summary.twr * 100).toFixed(2)}%` : "--"}
                    tooltip={definitionTooltip(definitions, "twr")}
                    coverageStatus={metricCoverage("twr")}
                    asOf={asOf}
                    reasonCodes={metricReasonCodes("twr")}
                />
                <MetricCard
                    label="MWR (Personal)"
                    value={summary.mwr !== null ? `${(summary.mwr * 100).toFixed(2)}%` : "--"}
                    tooltip={definitionTooltip(definitions, "mwr")}
                    coverageStatus={metricCoverage("mwr")}
                    asOf={asOf}
                    reasonCodes={metricReasonCodes("mwr")}
                />
                <MetricCard
                    label="Total Return"
                    value={totalReturn !== null ? `${(totalReturn * 100).toFixed(2)}%` : "--"}
                    subtext="From first valuation"
                    coverageStatus={metricCoverage("total_return")}
                    asOf={asOf}
                    reasonCodes={metricReasonCodes("total_return")}
                />
            </div>

            {/* Portfolio Value */}
            <div className="bg-white border border-gray-200 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#0F172A] mb-4">Portfolio Value</h3>
                {performance.length === 0 ? (
                    <div className="text-sm text-gray-500">No performance data available.</div>
                ) : (
                    <ResponsiveContainer width="100%" height={320}>
                        <LineChart data={performanceSeries}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#E3E7EE" />
                            <XAxis dataKey="date" stroke="#64748B" style={{ fontSize: "12px" }} />
                            <YAxis stroke="#64748B" style={{ fontSize: "12px" }} />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: "white",
                                    border: "1px solid #E3E7EE",
                                    borderRadius: "0.5rem",
                                }}
                                formatter={(value) =>
                                    value != null ? [`$${Number(value).toLocaleString()}`, "Value"] : ["--", "Value"]
                                }
                            />
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

            <details className="bg-white border border-gray-200 rounded-lg p-6">
                <summary className="cursor-pointer text-lg font-semibold text-[#0F172A] mb-4">Drawdowns</summary>
                {performance.length === 0 ? (
                    <div className="text-sm text-gray-500">No drawdown data available.</div>
                ) : (
                    <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={performanceSeries}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#E3E7EE" />
                            <XAxis dataKey="date" stroke="#64748B" style={{ fontSize: "12px" }} />
                            <YAxis
                                stroke="#64748B"
                                style={{ fontSize: "12px" }}
                                tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: "white",
                                    border: "1px solid #E3E7EE",
                                    borderRadius: "0.5rem",
                                }}
                                formatter={(value) =>
                                    value != null ? [`${(Number(value) * 100).toFixed(2)}%`, "Drawdown"] : ["--", "Drawdown"]
                                }
                            />
                            <Line
                                type="monotone"
                                dataKey="drawdown"
                                stroke="#64748B"
                                strokeWidth={2}
                                dot={false}
                                name="Drawdown"
                            />
                        </LineChart>
                    </ResponsiveContainer>
                )}
            </details>

            <details className="bg-white border border-gray-200 rounded-lg p-6">
                <summary className="cursor-pointer text-lg font-semibold text-[#0F172A] mb-4">Monthly Returns</summary>
                {monthly_returns.length === 0 ? (
                    <div className="text-sm text-gray-500">No monthly return data available.</div>
                ) : (
                    <ResponsiveContainer width="100%" height={260}>
                        <BarChart data={monthly_returns}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#E3E7EE" />
                            <XAxis dataKey="date" stroke="#64748B" style={{ fontSize: "12px" }} />
                            <YAxis
                                stroke="#64748B"
                                style={{ fontSize: "12px" }}
                                tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: "white",
                                    border: "1px solid #E3E7EE",
                                    borderRadius: "0.5rem",
                                }}
                                formatter={(value) =>
                                    value != null ? [`${(Number(value) * 100).toFixed(2)}%`, "Return"] : ["--", "Return"]
                                }
                            />
                            <Bar dataKey="return" fill="#2563EB" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                )}
            </details>

            <details className="bg-white border border-gray-200 rounded-lg p-6">
                <summary className="cursor-pointer text-lg font-semibold text-[#0F172A] mb-4">Return Drivers</summary>
                {!hasAttribution ? (
                    <div className="text-sm text-gray-500">No attribution data available.</div>
                ) : (
                    <div className="space-y-4">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <MetricCard
                                label="Allocation Effect"
                                value={attribution_summary.allocation !== null ? `${(attribution_summary.allocation * 100).toFixed(2)}%` : "--"}
                                tooltip={definitionTooltip(definitions, "allocation_effect")}
                                coverageStatus={metricCoverage("allocation_effect")}
                                asOf={asOf}
                                reasonCodes={metricReasonCodes("allocation_effect")}
                            />
                            <MetricCard
                                label="Selection Effect"
                                value={attribution_summary.selection !== null ? `${(attribution_summary.selection * 100).toFixed(2)}%` : "--"}
                                tooltip={definitionTooltip(definitions, "selection_effect")}
                                coverageStatus={metricCoverage("selection_effect")}
                                asOf={asOf}
                                reasonCodes={metricReasonCodes("selection_effect")}
                            />
                            <MetricCard
                                label="Interaction Effect"
                                value={attribution_summary.interaction !== null ? `${(attribution_summary.interaction * 100).toFixed(2)}%` : "--"}
                                tooltip={definitionTooltip(definitions, "interaction_effect")}
                                coverageStatus={metricCoverage("interaction_effect")}
                                asOf={asOf}
                                reasonCodes={metricReasonCodes("interaction_effect")}
                            />
                        </div>
                        {attribution_timeseries && attribution_timeseries.length > 0 ? (
                            <ResponsiveContainer width="100%" height={220}>
                                <LineChart data={attribution_timeseries}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#E3E7EE" />
                                    <XAxis dataKey="date" stroke="#64748B" style={{ fontSize: "12px" }} />
                                    <YAxis
                                        stroke="#64748B"
                                        style={{ fontSize: "12px" }}
                                        tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: "white",
                                            border: "1px solid #E3E7EE",
                                            borderRadius: "0.5rem",
                                        }}
                                        formatter={(value) =>
                                            value != null ? [`${(Number(value) * 100).toFixed(2)}%`, "Return"] : ["--", "Return"]
                                        }
                                    />
                                    <Line type="monotone" dataKey="allocation" stroke="#2563EB" strokeWidth={2} dot={false} />
                                    <Line type="monotone" dataKey="selection" stroke="#16A34A" strokeWidth={2} dot={false} />
                                    <Line type="monotone" dataKey="interaction" stroke="#F97316" strokeWidth={2} dot={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="text-sm text-gray-500">No attribution timeseries available.</div>
                        )}
                        {attributionRows.length > 0 && (
                            <div className="border border-gray-200 rounded-lg overflow-hidden">
                                <div className="bg-gray-50 px-4 py-2 text-sm text-gray-600">Top Contributors</div>
                                <div className="divide-y divide-gray-200">
                                    {attributionRows.map((row) => (
                                        <div key={row.ticker} className="px-4 py-2 flex items-center justify-between text-sm">
                                            <span className="font-medium text-gray-700">{row.ticker}</span>
                                            <span className="text-gray-500">{(row.total * 100).toFixed(2)}%</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </details>

            <details className="bg-white border border-gray-200 rounded-lg p-6">
                <summary className="cursor-pointer text-lg font-semibold text-[#0F172A] mb-4">Macro Context</summary>
                {macroStatus === "unavailable" ? (
                    <div className="text-sm text-gray-500">
                        Macro context unavailable.
                        {macro?.warnings && macro.warnings.length > 0 && (
                            <div className="mt-2 text-xs text-gray-400">
                                {macro.warnings.join(" ")}
                            </div>
                        )}
                    </div>
                ) : !latestMacro ? (
                    <div className="text-sm text-gray-500">No macro regime data available.</div>
                ) : (
                    <div className="space-y-3">
                        {macroStatus === "partial" && (
                            <div className="text-xs text-gray-400">
                                Macro context limited. Some tags may be suppressed.
                            </div>
                        )}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <div className="border border-gray-200 rounded-lg p-3">
                            <div className="text-gray-500">Inflation YoY</div>
                            <div className="text-lg font-semibold text-[#0F172A]">
                                {latestMacro.inflation_yoy != null ? `${(latestMacro.inflation_yoy * 100).toFixed(2)}%` : "--"}
                            </div>
                            <div className="text-xs text-gray-400 mt-1">
                                {Boolean(latestMacro.high_inflation) ? "High inflation regime" : "Inflation stable"}
                            </div>
                        </div>
                        <div className="border border-gray-200 rounded-lg p-3">
                            <div className="text-gray-500">Rates</div>
                            <div className="text-lg font-semibold text-[#0F172A]">
                                {latestMacro.fed_funds != null ? `${latestMacro.fed_funds.toFixed(2)}%` : "--"}
                            </div>
                            <div className="text-xs text-gray-400 mt-1">
                                {Boolean(latestMacro.rising_rates) ? "Rising rate regime" : "Rates stable"}
                            </div>
                        </div>
                        <div className="border border-gray-200 rounded-lg p-3">
                            <div className="text-gray-500">Risk Tone (VIX)</div>
                            <div className="text-lg font-semibold text-[#0F172A]">
                                {latestMacro.vix != null ? latestMacro.vix.toFixed(1) : "--"}
                            </div>
                            <div className="text-xs text-gray-400 mt-1">
                                {Boolean(latestMacro.risk_off) ? "Risk-off regime" : "Risk-on regime"}
                            </div>
                        </div>
                        </div>
                    </div>
                )}
            </details>

            <details className="bg-white border border-gray-200 rounded-lg p-6">
                <summary className="cursor-pointer text-lg font-semibold text-[#0F172A] mb-4">Portfolio vs Benchmark</summary>
                {!hasBenchmarkComparison ? (
                    <div className="text-sm text-gray-500">No benchmark comparison data available.</div>
                ) : (
                    <div className="space-y-4">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <MetricCard
                                label="Tracking Error"
                                value={
                                    benchmark_comparison.tracking_error != null
                                        ? `${(benchmark_comparison.tracking_error * 100).toFixed(2)}%`
                                        : "--"
                                }
                                tooltip={definitionTooltip(definitions, "tracking_error")}
                                coverageStatus={metricCoverage("tracking_error")}
                                asOf={asOf}
                                reasonCodes={metricReasonCodes("tracking_error")}
                            />
                            <MetricCard
                                label="Correlation"
                                value={
                                    benchmark_comparison.correlation != null
                                        ? benchmark_comparison.correlation.toFixed(2)
                                        : "--"
                                }
                                tooltip={definitionTooltip(definitions, "benchmark_correlation")}
                                coverageStatus={metricCoverage("benchmark_correlation")}
                                asOf={asOf}
                                reasonCodes={metricReasonCodes("benchmark_correlation")}
                            />
                            <MetricCard
                                label="Benchmark Volatility"
                                value={
                                    benchmark_comparison.benchmark_volatility != null
                                        ? `${(benchmark_comparison.benchmark_volatility * 100).toFixed(2)}%`
                                        : "--"
                                }
                                tooltip={definitionTooltip(definitions, "benchmark_volatility")}
                                coverageStatus={metricCoverage("benchmark_volatility")}
                                asOf={asOf}
                                reasonCodes={metricReasonCodes("benchmark_volatility")}
                            />
                        </div>
                        {benchmark_timeseries && benchmark_timeseries.length > 0 ? (
                            <ResponsiveContainer width="100%" height={220}>
                                <LineChart data={benchmark_timeseries}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#E3E7EE" />
                                    <XAxis dataKey="date" stroke="#64748B" style={{ fontSize: "12px" }} />
                                    <YAxis
                                        stroke="#64748B"
                                        style={{ fontSize: "12px" }}
                                        tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: "white",
                                            border: "1px solid #E3E7EE",
                                            borderRadius: "0.5rem",
                                        }}
                                        formatter={(value) =>
                                            value != null ? [`${(Number(value) * 100).toFixed(2)}%`, "Return"] : ["--", "Return"]
                                        }
                                    />
                                    <Line type="monotone" dataKey="active_return" stroke="#2563EB" strokeWidth={2} dot={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="text-sm text-gray-500">No benchmark timeseries available.</div>
                        )}
                    </div>
                )}
            </details>
        </div>
    );
}
