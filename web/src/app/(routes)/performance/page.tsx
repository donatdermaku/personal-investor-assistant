"use client";

import { MetricCard } from "@/components/ui/MetricCard";
import { definitionTooltip } from "@/lib/definitions";
import { useNexus } from "@/components/nexus/NexusProvider";
import { EmptyState } from "@/components/nexus/EmptyState";
import { SkeletonCard, SkeletonBlock } from "@/components/nexus/Skeleton";
import { SectionContext } from "@/components/nexus/SectionContext";
import { BentoGrid, BentoItem } from "@/components/nexus/BentoGrid";
import { getMetricReasons, getMetricStatus } from "@/lib/coverageLogic";
import { LineChart, Line, Area, AreaChart, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts";

export default function PerformancePage() {
    const { state, status, error, mode, setMode, openRunCreator } = useNexus();

    if (status === "error") {
        return (
            <EmptyState
                title="Unable to load performance data"
                description={error || "Check API connectivity, then retry in Live Mode or continue in Demo Mode."}
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
                    <SkeletonBlock className="h-8 w-40 bg-[var(--color-nexus-surface)]" />
                    <SkeletonBlock className="mt-2 h-4 w-56 bg-[var(--color-nexus-surface)]" />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <SkeletonCard />
                    <SkeletonCard />
                    <SkeletonCard />
                </div>
                <div className="nexus-card p-6 h-80 flex flex-col">
                    <SkeletonBlock className="h-5 w-32 bg-[var(--color-nexus-surface-hover)]" />
                    <SkeletonBlock className="mt-4 flex-1 w-full bg-[var(--color-nexus-surface-hover)]" />
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
        <div className="space-y-8 animate-in fade-in duration-500">
            <div>
                <h2 className="text-3xl font-sans font-bold text-[var(--color-nexus-text-primary)] mb-2 tracking-tight">Performance</h2>
                <p className="text-[var(--color-nexus-text-secondary)]">
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
                ]}
            />

            {summary.errors && summary.errors.length > 0 && (
                <div className="rounded-none border border-[var(--color-nexus-warning)] bg-[var(--color-nexus-warning)]/10 px-4 py-3 text-sm text-[var(--color-nexus-warning)] font-mono">
                    Some inputs are missing or incomplete for this run. Metrics relying on those inputs may be limited.
                </div>
            )}

            <BentoGrid>
                {/* HEADLINE METRICS - Span 12 */}
                <BentoItem span={12} className="p-0 border-none bg-transparent hover:shadow-none hover:border-transparent">
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
                </BentoItem>

                {/* MAIN CHART: Portfolio Value - Span 8 */}
                <BentoItem span={8} title="Portfolio Value" rowSpan={2}>
                    {performance.length === 0 ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">No performance data available.</div>
                    ) : (
                        <div className="flex-1 min-h-0 h-72 md:h-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={performanceSeries}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-nexus-border)" vertical={false} />
                                    <XAxis
                                        dataKey="date"
                                        stroke="var(--color-nexus-text-muted)"
                                        style={{ fontSize: "10px", fontFamily: "var(--font-mono)" }}
                                        tickLine={false}
                                        axisLine={false}
                                        dy={10}
                                    />
                                    <YAxis
                                        stroke="var(--color-nexus-text-muted)"
                                        style={{ fontSize: "10px", fontFamily: "var(--font-mono)" }}
                                        tickLine={false}
                                        axisLine={false}
                                        dx={-10}
                                        domain={["auto", "auto"]}
                                    />
                                    <Tooltip
                                        animationDuration={100}
                                        contentStyle={{
                                            backgroundColor: "var(--color-nexus-surface)",
                                            borderColor: "var(--color-nexus-primary)",
                                            borderRadius: "0",
                                            color: "var(--color-nexus-text-primary)",
                                            fontFamily: "var(--font-mono)",
                                            fontSize: "12px"
                                        }}
                                        itemStyle={{ color: "var(--color-nexus-text-primary)" }}
                                        formatter={(value) =>
                                            value != null ? [`$${Number(value).toLocaleString()}`, "Value"] : ["--", "Value"]
                                        }
                                        labelStyle={{ color: "var(--color-nexus-text-secondary)", marginBottom: "0.5rem" }}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="value"
                                        stroke="var(--color-nexus-primary)"
                                        strokeWidth={2}
                                        fill="var(--color-nexus-primary)"
                                        fillOpacity={0.05}
                                        name="Portfolio Value"
                                        activeDot={{ r: 4, strokeWidth: 0, fill: "var(--color-nexus-text-primary)" }}
                                    />
                                    {hasBenchmark && (
                                        <Area
                                            type="monotone"
                                            dataKey="benchmark"
                                            stroke="var(--color-nexus-secondary)"
                                            strokeWidth={1}
                                            strokeDasharray="4 4"
                                            fill="transparent"
                                            dot={false}
                                            name="Benchmark"
                                            activeDot={false}
                                        />
                                    )}
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </BentoItem>

                {/* SECONDARY CHART 1: Drawdowns - Span 4 */}
                <BentoItem span={4} title="Drawdowns" rowSpan={2}>
                    {performance.length === 0 ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">No drawdown data available.</div>
                    ) : (
                        <div className="flex-1 min-h-0 h-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={performanceSeries}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-nexus-border)" vertical={false} />
                                    <XAxis dataKey="date" hide />
                                    <YAxis
                                        stroke="var(--color-nexus-text-muted)"
                                        style={{ fontSize: "10px", fontFamily: "var(--font-mono)" }}
                                        tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`}
                                        tickLine={false} axisLine={false} dx={-10}
                                        width={30}
                                    />
                                    <Tooltip
                                        animationDuration={100}
                                        contentStyle={{
                                            backgroundColor: "var(--color-nexus-surface)",
                                            borderColor: "var(--color-nexus-danger)",
                                            borderRadius: "0",
                                            color: "var(--color-nexus-text-primary)",
                                            fontFamily: "var(--font-mono)",
                                            fontSize: "12px"
                                        }}
                                        itemStyle={{ color: "var(--color-nexus-danger)" }}
                                        formatter={(value) =>
                                            value != null ? [`${(Number(value) * 100).toFixed(2)}%`, "Drawdown"] : ["--", "Drawdown"]
                                        }
                                        labelStyle={{ color: "var(--color-nexus-text-secondary)", marginBottom: "0.5rem" }}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="drawdown"
                                        stroke="var(--color-nexus-danger)"
                                        strokeWidth={1}
                                        fill="var(--color-nexus-danger)"
                                        fillOpacity={0.1}
                                        activeDot={{ r: 4, strokeWidth: 0, fill: "var(--color-nexus-danger)" }}
                                        name="Drawdown"
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </BentoItem>

                {/* SECONDARY CHART 2: Monthly Returns - Span 4 */}
                <BentoItem span={4} title="Monthly Returns" rowSpan={2}>
                    {monthly_returns.length === 0 ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">No monthly return data available.</div>
                    ) : (
                        <div className="flex-1 min-h-0 h-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={monthly_returns}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-nexus-border)" vertical={false} />
                                    <XAxis dataKey="date" hide />
                                    <YAxis
                                        stroke="var(--color-nexus-text-muted)"
                                        style={{ fontSize: "10px", fontFamily: "var(--font-mono)" }}
                                        tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`}
                                        tickLine={false} axisLine={false} dx={-10}
                                        width={30}
                                    />
                                    <Tooltip
                                        animationDuration={100}
                                        cursor={{ fill: "var(--color-nexus-surface-hover)" }}
                                        contentStyle={{
                                            backgroundColor: "var(--color-nexus-surface)",
                                            borderColor: "var(--color-nexus-primary)",
                                            borderRadius: "0",
                                            color: "var(--color-nexus-text-primary)",
                                            fontFamily: "var(--font-mono)",
                                            fontSize: "12px"
                                        }}
                                        formatter={(value) =>
                                            value != null ? [`${(Number(value) * 100).toFixed(2)}%`, "Return"] : ["--", "Return"]
                                        }
                                        labelStyle={{ color: "var(--color-nexus-text-secondary)", marginBottom: "0.5rem" }}
                                    />
                                    <Bar dataKey="return" fill="var(--color-nexus-primary)" radius={[0, 0, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </BentoItem>

                {/* ATTRIBUTION & MACRO - Span 12 (Bottom strip) */}
                <BentoItem span={6} title="Attribution & Drivers" rowSpan={2}>
                    {!hasAttribution ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">No attribution data available.</div>
                    ) : (
                        <div className="space-y-4 h-full flex flex-col">
                            {attribution_timeseries && attribution_timeseries.length > 0 ? (
                                <div className="flex-1 min-h-0">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={attribution_timeseries}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-nexus-border)" vertical={false} />
                                            <XAxis dataKey="date" hide />
                                            <YAxis
                                                stroke="var(--color-nexus-text-muted)"
                                                style={{ fontSize: "10px", fontFamily: "var(--font-mono)" }}
                                                tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`}
                                                tickLine={false} axisLine={false} dx={-10}
                                                width={30}
                                            />
                                            <Tooltip
                                                animationDuration={100}
                                                contentStyle={{
                                                    backgroundColor: "var(--color-nexus-surface)",
                                                    borderColor: "var(--color-nexus-primary)",
                                                    borderRadius: "0",
                                                    color: "var(--color-nexus-text-primary)",
                                                    fontFamily: "var(--font-mono)",
                                                    fontSize: "12px"
                                                }}
                                                formatter={(value) =>
                                                    value != null ? [`${(Number(value) * 100).toFixed(2)}%`, "Return"] : ["--", "Return"]
                                                }
                                                labelStyle={{ color: "var(--color-nexus-text-secondary)", marginBottom: "0.5rem" }}
                                            />
                                            <Area type="monotone" dataKey="allocation" stackId="1" stroke="var(--color-nexus-primary)" fill="var(--color-nexus-primary)" fillOpacity={0.6} name="Allocation" />
                                            <Area type="monotone" dataKey="selection" stackId="1" stroke="var(--color-nexus-success)" fill="var(--color-nexus-success)" fillOpacity={0.6} name="Selection" />
                                            <Area type="monotone" dataKey="interaction" stackId="1" stroke="var(--color-nexus-accent)" fill="var(--color-nexus-accent)" fillOpacity={0.6} name="Interaction" />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            ) : null}

                            <div className="grid grid-cols-3 gap-2 text-xs mt-auto">
                                <div>
                                    <span className="block text-[var(--color-nexus-text-muted)]">Allocation</span>
                                    <span className="font-mono text-[var(--color-nexus-text-primary)]">
                                        {attribution_summary.allocation !== null ? `${(attribution_summary.allocation * 100).toFixed(2)}%` : "--"}
                                    </span>
                                </div>
                                <div>
                                    <span className="block text-[var(--color-nexus-text-muted)]">Selection</span>
                                    <span className="font-mono text-[var(--color-nexus-text-primary)]">
                                        {attribution_summary.selection !== null ? `${(attribution_summary.selection * 100).toFixed(2)}%` : "--"}
                                    </span>
                                </div>
                                <div>
                                    <span className="block text-[var(--color-nexus-text-muted)]">Interaction</span>
                                    <span className="font-mono text-[var(--color-nexus-text-primary)]">
                                        {attribution_summary.interaction !== null ? `${(attribution_summary.interaction * 100).toFixed(2)}%` : "--"}
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}
                </BentoItem>

                <BentoItem span={6} title="Macro Context" rowSpan={2}>
                    {macroStatus === "unavailable" ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">
                            Macro context unavailable.
                        </div>
                    ) : !latestMacro ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">No macro regime data available.</div>
                    ) : (
                        <div className="grid grid-cols-3 gap-4 h-full">
                            <div className="flex flex-col h-full">
                                <div className="text-[var(--color-nexus-text-secondary)] text-[10px] uppercase tracking-wider mb-1">Inflation</div>
                                <div className="text-xl font-bold text-[var(--color-nexus-text-primary)] font-mono">
                                    {latestMacro.inflation_yoy != null ? `${(latestMacro.inflation_yoy * 100).toFixed(2)}%` : "--"}
                                </div>
                                <div className="flex-1 min-h-[40px] mt-2">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={macro_regimes}>
                                            <Line type="monotone" dataKey="inflation_yoy" stroke="var(--color-nexus-warning)" strokeWidth={1.5} dot={false} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                            <div className="flex flex-col h-full border-l border-r border-[var(--color-nexus-border)] px-4">
                                <div className="text-[var(--color-nexus-text-secondary)] text-[10px] uppercase tracking-wider mb-1">Rates</div>
                                <div className="text-xl font-bold text-[var(--color-nexus-text-primary)] font-mono">
                                    {latestMacro.fed_funds != null ? `${latestMacro.fed_funds.toFixed(2)}%` : "--"}
                                </div>
                                <div className="flex-1 min-h-[40px] mt-2">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={macro_regimes}>
                                            <Line type="monotone" dataKey="fed_funds" stroke="var(--color-nexus-text-primary)" strokeWidth={1.5} dot={false} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                            <div className="flex flex-col h-full">
                                <div className="text-[var(--color-nexus-text-secondary)] text-[10px] uppercase tracking-wider mb-1">VIX</div>
                                <div className="text-xl font-bold text-[var(--color-nexus-text-primary)] font-mono">
                                    {latestMacro.vix != null ? latestMacro.vix.toFixed(1) : "--"}
                                </div>
                                <div className="flex-1 min-h-[40px] mt-2">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={macro_regimes}>
                                            <Line type="monotone" dataKey="vix" stroke="var(--color-nexus-primary)" strokeWidth={1.5} dot={false} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>
                    )}
                </BentoItem>
            </BentoGrid>
        </div>
    );
}
