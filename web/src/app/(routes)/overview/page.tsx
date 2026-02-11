"use client";

import { useNexus } from "@/components/nexus/NexusProvider";
import { MetricCard } from "@/components/ui/MetricCard";
import { definitionTooltip } from "@/lib/definitions";
import { EmptyState } from "@/components/nexus/EmptyState";
import { SkeletonCard, SkeletonBlock } from "@/components/nexus/Skeleton";
import { SectionContext } from "@/components/nexus/SectionContext";
import { BentoGrid, BentoItem } from "@/components/nexus/BentoGrid";
import { getMetricReasons, getMetricStatus } from "@/lib/coverageLogic";
import { LineChart, Line, Area, AreaChart, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export default function OverviewPage() {
    const { state, status, error, mode, setMode, openRunCreator } = useNexus();

    if (status === "error") {
        return (
            <EmptyState
                title="Unable to load dashboard"
                description={error || "Check API connectivity, then retry in Live Mode or continue in Demo Mode."}
                primaryAction={{ label: "Retry in Live Mode", onClick: () => setMode("live") }}
                secondaryAction={{ label: "Switch to Demo Mode", onClick: () => setMode("demo") }}
            />
        );
    }

    if (status === "empty") {
        return (
            <EmptyState
                title="Welcome to Nexus"
                description="Your personal investment assistant is ready. Create a run to see your portfolio analytics."
                primaryAction={{ label: "Create First Run", onClick: openRunCreator }}
                secondaryAction={{ label: "Switch to Demo Mode", onClick: () => setMode("demo") }}
            />
        );
    }

    if (status === "loading" || !state) {
        return (
            <div className="space-y-8">
                <div>
                    <SkeletonBlock className="h-8 w-48 bg-[var(--color-nexus-surface)]" />
                    <SkeletonBlock className="mt-2 h-4 w-64 bg-[var(--color-nexus-surface)]" />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <SkeletonCard />
                    <SkeletonCard />
                    <SkeletonCard />
                    <SkeletonCard />
                </div>
                <div className="nexus-card p-6 h-96 flex flex-col">
                    <SkeletonBlock className="h-6 w-32 bg-[var(--color-nexus-surface-hover)]" />
                    <SkeletonBlock className="mt-6 flex-1 w-full bg-[var(--color-nexus-surface-hover)]" />
                </div>
            </div>
        );
    }

    const { summary, performance, risk, definitions, diagnostics, manifest, coverage_summary } = state;
    const performanceSeries = performance as Array<{ date: string; value: number | null }>;
    const lastValue = performance[performance.length - 1]?.value ?? null;
    const metricStatus = (kpiKey: string) => getMetricStatus(kpiKey, coverage_summary ?? null);
    const metricReasons = (kpiKey: string) => getMetricReasons(kpiKey, coverage_summary ?? null);
    const metricCoverage = (kpiKey: string) => (metricStatus(kpiKey) === "insufficient" ? "insufficient" : undefined);
    const metricReasonCodes = (kpiKey: string) =>
        metricStatus(kpiKey) === "insufficient" ? metricReasons(kpiKey) : undefined;
    const asOf = summary.last_date || manifest.timestamp;

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            <div>
                <h2 className="text-3xl font-sans font-bold text-[var(--color-nexus-text-primary)] mb-2 tracking-tight">Overview</h2>
                <p className="text-[var(--color-nexus-text-secondary)]">
                    {mode === "demo" ? "Demo portfolio performance summary" : "Portfolio performance summary"}
                </p>
            </div>

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
                ]}
            />

            {manifest.new_run_created && (
                <div className="bg-[var(--color-nexus-success)]/10 border border-[var(--color-nexus-success)]/30 text-[var(--color-nexus-success)] px-4 py-3 rounded-none flex items-center justify-between animate-in slide-in-from-top-2 duration-300">
                    <span className="font-mono text-sm">New run created successfully.</span>
                </div>
            )}

            <BentoGrid>
                {/* TOP ROW: KEY METRICS - 4 Cards */}
                <BentoItem span={3} title="TWR (YTD)">
                    <div className="flex flex-col justify-end h-full">
                        <span className={`text-2xl font-mono font-bold ${summary.twr && summary.twr >= 0 ? "text-[var(--color-nexus-success)]" : "text-[var(--color-nexus-danger)]"}`}>
                            {summary.twr !== null ? `${(summary.twr * 100).toFixed(2)}%` : "--"}
                        </span>
                        <span className="text-xs text-[var(--color-nexus-text-muted)] mt-1">Time-Weighted Return</span>
                    </div>
                </BentoItem>
                <BentoItem span={3} title="MWR (YTD)">
                    <div className="flex flex-col justify-end h-full">
                        <span className={`text-2xl font-mono font-bold ${summary.mwr && summary.mwr >= 0 ? "text-[var(--color-nexus-success)]" : "text-[var(--color-nexus-danger)]"}`}>
                            {summary.mwr !== null ? `${(summary.mwr * 100).toFixed(2)}%` : "--"}
                        </span>
                        <span className="text-xs text-[var(--color-nexus-text-muted)] mt-1">Money-Weighted Return</span>
                    </div>
                </BentoItem>
                <BentoItem span={3} title="Portfolio Value">
                    <div className="flex flex-col justify-end h-full">
                        <span className="text-2xl font-mono font-bold text-[var(--color-nexus-text-primary)]">
                            {lastValue !== null ? `$${lastValue.toLocaleString()}` : "--"}
                        </span>
                        <span className="text-xs text-[var(--color-nexus-text-muted)] mt-1">Current Balance</span>
                    </div>
                </BentoItem>
                <BentoItem span={3} title="Max Drawdown">
                    <div className="flex flex-col justify-end h-full">
                        <span className="text-2xl font-mono font-bold text-[var(--color-nexus-danger)]">
                            {summary.max_drawdown !== null ? `${(summary.max_drawdown * 100).toFixed(2)}%` : "--"}
                        </span>
                        <span className="text-xs text-[var(--color-nexus-text-muted)] mt-1">Peak to Trough</span>
                    </div>
                </BentoItem>

                {/* HERO ROW: EQUITY CURVE + QUICK STATS */}
                <BentoItem span={8} title="Equity Curve" rowSpan={2}>
                    {performance.length === 0 ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">No equity curve data available.</div>
                    ) : (
                        <div className="flex-1 min-h-0 h-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={performanceSeries}>
                                    <defs>
                                        <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="var(--color-nexus-primary)" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="var(--color-nexus-primary)" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-nexus-border)" vertical={false} />
                                    <XAxis
                                        dataKey="date"
                                        stroke="var(--color-nexus-text-muted)"
                                        style={{ fontSize: "10px", fontFamily: "var(--font-mono)" }}
                                        tickLine={false}
                                        axisLine={false}
                                        dy={10}
                                        minTickGap={30}
                                    />
                                    <YAxis
                                        stroke="var(--color-nexus-text-muted)"
                                        style={{ fontSize: "10px", fontFamily: "var(--font-mono)" }}
                                        tickLine={false}
                                        axisLine={false}
                                        dx={-10}
                                        domain={["auto", "auto"]}
                                        tickFormatter={(val) => `$${val / 1000}k`}
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
                                        fill="url(#colorValue)"
                                        activeDot={{ r: 4, strokeWidth: 0, fill: "var(--color-nexus-text-primary)" }}
                                        name="Portfolio Value"
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </BentoItem>

                {/* QUICK STATS - Span 4 */}
                <BentoItem span={4} title="Quick Stats" rowSpan={1}>
                    <div className="space-y-4">
                        <div className="flex justify-between items-center py-2 border-b border-[var(--color-nexus-border)]">
                            <span className="text-[var(--color-nexus-text-secondary)] text-sm">Volatility (30d)</span>
                            <span className="font-mono text-[var(--color-nexus-text-primary)]">
                                {risk.volatility != null ? `${(risk.volatility * 100).toFixed(2)}%` : "--"}
                            </span>
                        </div>
                        <div className="flex justify-between items-center py-2 border-b border-[var(--color-nexus-border)]">
                            <span className="text-[var(--color-nexus-text-secondary)] text-sm">Sharpe Ratio</span>
                            <span className="font-mono text-[var(--color-nexus-text-primary)]">
                                {risk.sharpe != null ? risk.sharpe.toFixed(2) : "--"}
                            </span>
                        </div>
                        <div className="flex justify-between items-center py-2 border-b border-[var(--color-nexus-border)]">
                            <span className="text-[var(--color-nexus-text-secondary)] text-sm">Beta to SPY</span>
                            <span className="font-mono text-[var(--color-nexus-text-primary)]">
                                {risk.beta != null ? risk.beta.toFixed(2) : "--"}
                            </span>
                        </div>
                    </div>
                </BentoItem>

                {/* DIAGNOSTICS - Span 4 (Using explicit span 4 to fill the gap next to Equity Curve which is rowSpan 2? No wait.)
                   Equity Curve is rowSpan 2 (height ~ 2 * 12rem + gap = 24rem + gap).
                   Quick Stats is rowSpan 1.
                   We need another item of rowSpan 1 to stack under Quick Stats to fill the column.
                   Let's put Diagnostics THERE.
                */}
                <BentoItem span={4} title="System Diagnostics" rowSpan={1}>
                    {!diagnostics || diagnostics.length === 0 ? (
                        <div className="text-sm text-[var(--color-nexus-success)] flex items-center gap-2 h-full">
                            <span>✓ All systems nominal.</span>
                        </div>
                    ) : (
                        <div className="space-y-2 overflow-y-auto max-h-[140px]">
                            {diagnostics.slice(0, 3).map((diag, idx) => (
                                <div key={idx} className="flex items-start space-x-2 p-2 bg-[var(--color-nexus-surface-hover)]/50 rounded border border-[var(--color-nexus-border)]">
                                    <span className={`
                                        flex-shrink-0 w-1.5 h-1.5 mt-1.5 rounded-full 
                                        ${diag.severity === "critical" ? "bg-[var(--color-nexus-danger)]" : (diag.severity === "high" ? "bg-[var(--color-nexus-danger)]" : "bg-[var(--color-nexus-warning)]")}
                                    `} />
                                    <div className="flex-1 min-w-0">
                                        <div className="text-xs font-medium text-[var(--color-nexus-text-primary)] truncate">
                                            {diag.check}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </BentoItem>

            </BentoGrid>
        </div>
    );
}
