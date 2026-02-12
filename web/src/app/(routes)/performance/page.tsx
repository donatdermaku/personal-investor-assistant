"use client";

import { useNexus } from "@/components/nexus/NexusProvider";
import { EmptyState } from "@/components/nexus/EmptyState";
import { SkeletonBlock } from "@/components/nexus/Skeleton";
import { SectionContext } from "@/components/nexus/SectionContext";
import { LazyChart } from "@/components/nexus/LazyChart";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { downsampleSeries } from "@/lib/chartPerformance";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function PerformancePage() {
    const { state, status, error, mode, setMode, openRunCreator } = useNexus();
    const isPhone = useMediaQuery("(max-width: 767px)");

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
            <div className="nexus-page">
                <div>
                    <SkeletonBlock className="h-8 w-40 bg-[var(--color-nexus-surface)]" />
                    <SkeletonBlock className="mt-2 h-4 w-56 bg-[var(--color-nexus-surface)]" />
                </div>
                <div className="nexus-metrics-grid metrics-3">
                    <SkeletonBlock className="h-28 bg-[var(--color-nexus-surface)]" />
                    <SkeletonBlock className="h-28 bg-[var(--color-nexus-surface)]" />
                    <SkeletonBlock className="h-28 bg-[var(--color-nexus-surface)]" />
                </div>
            </div>
        );
    }

    const { summary, performance, monthly_returns, attribution_summary, attribution_timeseries } = state;
    const firstValue = performance[0]?.value ?? null;
    const lastValue = performance[performance.length - 1]?.value ?? null;
    const latestDrawdown = performance[performance.length - 1]?.drawdown ?? null;
    const totalReturn = firstValue && lastValue ? (lastValue / firstValue) - 1 : null;
    const hasAttribution = attribution_summary && typeof attribution_summary.allocation === "number";
    const attributionBars = attribution_summary?.per_asset?.slice(0, 8) ?? [];
    const lightPerformance = downsampleSeries(performance, isPhone ? 120 : 260);
    const lightMonthly = downsampleSeries(monthly_returns, isPhone ? 60 : 120);
    const lightAttributionSeries = downsampleSeries(attribution_timeseries ?? [], isPhone ? 90 : 180);

    return (
        <div className="nexus-page animate-in fade-in duration-500">
            <div>
                <h2 className="nexus-heading-1 font-sans font-bold text-[var(--color-nexus-text-primary)] mb-2 tracking-tight">Performance</h2>
                <p className="text-[var(--color-nexus-text-secondary)] text-base">
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

            <section className="nexus-metrics-grid metrics-3">
                <article className="nexus-card p-6">
                    <div className="text-label mb-3">TWR</div>
                    <div className={`nexus-number font-mono font-bold ${summary.twr && summary.twr >= 0 ? "text-[var(--color-nexus-success)]" : "text-[var(--color-nexus-danger)]"}`}>
                        {summary.twr !== null ? `${(summary.twr * 100).toFixed(2)}%` : "--"}
                    </div>
                </article>
                <article className="nexus-card p-6">
                    <div className="text-label mb-3">MWR</div>
                    <div className={`nexus-number font-mono font-bold ${summary.mwr && summary.mwr >= 0 ? "text-[var(--color-nexus-success)]" : "text-[var(--color-nexus-danger)]"}`}>
                        {summary.mwr !== null ? `${(summary.mwr * 100).toFixed(2)}%` : "--"}
                    </div>
                </article>
                <article className="nexus-card p-6">
                    <div className="text-label mb-3">Total Return</div>
                    <div className={`nexus-number font-mono font-bold ${totalReturn && totalReturn >= 0 ? "text-[var(--color-nexus-success)]" : "text-[var(--color-nexus-danger)]"}`}>
                        {totalReturn !== null ? `${(totalReturn * 100).toFixed(2)}%` : "--"}
                    </div>
                </article>
            </section>

            <section className="grid gap-[var(--grid-gap)] min-[1200px]:grid-cols-5">
                <article className="nexus-card p-6 min-h-[330px] min-[1200px]:col-span-3">
                    <h3 className="nexus-heading-3 font-semibold text-[var(--color-nexus-text-primary)] mb-4">Portfolio Value</h3>
                    {lightPerformance.length === 0 ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">No performance data available.</div>
                    ) : (
                        <LazyChart heightClassName="h-[280px] min-[1200px]:h-[340px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={lightPerformance}>
                                    <defs>
                                        <linearGradient id="perfValue" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="var(--color-nexus-primary)" stopOpacity={0.22} />
                                            <stop offset="95%" stopColor="var(--color-nexus-primary)" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-nexus-border)" vertical={false} />
                                    <XAxis dataKey="date" tickLine={false} axisLine={false} minTickGap={28} />
                                    <YAxis tickLine={false} axisLine={false} />
                                    <Tooltip formatter={(value) => (value != null ? [`$${Number(value).toLocaleString()}`, "Value"] : ["--", "Value"])} />
                                    <Area type="monotone" dataKey="value" stroke="var(--color-nexus-primary)" fill="url(#perfValue)" strokeWidth={2} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </LazyChart>
                    )}
                </article>
                <article className="nexus-card p-6 min-h-[360px] min-[1200px]:col-span-2">
                    <h3 className="nexus-heading-3 font-semibold text-[var(--color-nexus-text-primary)] mb-4">Drawdowns</h3>
                    <div className="grid grid-cols-2 gap-3 mb-4">
                        <div className="border border-[var(--color-nexus-border)] bg-[var(--color-nexus-surface-hover)]/20 p-3">
                            <div className="text-[10px] uppercase tracking-wider text-[var(--color-nexus-text-muted)]">Current</div>
                            <div className="text-base font-mono text-[var(--color-nexus-danger)] nexus-inline-number">
                                {latestDrawdown != null ? `${(latestDrawdown * 100).toFixed(2)}%` : "--"}
                            </div>
                        </div>
                        <div className="border border-[var(--color-nexus-border)] bg-[var(--color-nexus-surface-hover)]/20 p-3">
                            <div className="text-[10px] uppercase tracking-wider text-[var(--color-nexus-text-muted)]">Max</div>
                            <div className="text-base font-mono text-[var(--color-nexus-danger)] nexus-inline-number">
                                {summary.max_drawdown != null ? `${(summary.max_drawdown * 100).toFixed(2)}%` : "--"}
                            </div>
                        </div>
                    </div>
                    {lightPerformance.length === 0 ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">No drawdown data available.</div>
                    ) : (
                        <LazyChart heightClassName="h-[290px] min-[1200px]:h-[340px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={lightPerformance}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-nexus-border)" vertical={false} />
                                    <XAxis dataKey="date" tickLine={false} axisLine={false} minTickGap={32} />
                                    <YAxis tickLine={false} axisLine={false} tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`} />
                                    <Tooltip formatter={(value) => (value != null ? [`${(Number(value) * 100).toFixed(2)}%`, "Drawdown"] : ["--", "Drawdown"])} />
                                    <Area type="monotone" dataKey="drawdown" stroke="var(--color-nexus-danger)" fill="var(--color-nexus-danger)" fillOpacity={0.15} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </LazyChart>
                    )}
                </article>
            </section>

            <section className="nexus-two-up">
                <article className="nexus-card p-6 min-h-[320px]">
                    <h3 className="nexus-heading-3 font-semibold text-[var(--color-nexus-text-primary)] mb-4">Monthly Returns</h3>
                    {monthly_returns.length === 0 ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">No monthly return data available.</div>
                    ) : (
                        <LazyChart heightClassName="h-[260px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={lightMonthly}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-nexus-border)" vertical={false} />
                                    <XAxis dataKey="date" hide />
                                    <YAxis tickLine={false} axisLine={false} tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`} />
                                    <Tooltip formatter={(value) => (value != null ? [`${(Number(value) * 100).toFixed(2)}%`, "Return"] : ["--", "Return"])} />
                                    <Bar dataKey="return" fill="var(--color-nexus-primary)" />
                                </BarChart>
                            </ResponsiveContainer>
                        </LazyChart>
                    )}
                </article>

                <article className="nexus-card p-6 min-h-[320px]">
                    <h3 className="nexus-heading-3 font-semibold text-[var(--color-nexus-text-primary)] mb-4">Attribution</h3>
                    {!hasAttribution ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">No attribution data available.</div>
                    ) : lightAttributionSeries.length > 0 ? (
                        <div className="space-y-4">
                            <LazyChart heightClassName="h-[180px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={lightAttributionSeries}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-nexus-border)" vertical={false} />
                                        <XAxis dataKey="date" hide />
                                        <YAxis tickLine={false} axisLine={false} tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`} />
                                        <Tooltip formatter={(value) => (value != null ? [`${(Number(value) * 100).toFixed(2)}%`, "Return"] : ["--", "Return"])} />
                                        <Area type="monotone" dataKey="allocation" stackId="1" stroke="var(--color-nexus-primary)" fill="var(--color-nexus-primary)" fillOpacity={0.65} />
                                        <Area type="monotone" dataKey="selection" stackId="1" stroke="var(--color-nexus-success)" fill="var(--color-nexus-success)" fillOpacity={0.65} />
                                        <Area type="monotone" dataKey="interaction" stackId="1" stroke="var(--color-nexus-accent)" fill="var(--color-nexus-accent)" fillOpacity={0.65} />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </LazyChart>
                            <div className="grid grid-cols-3 gap-3 text-xs">
                                <div>
                                    <div className="text-[var(--color-nexus-text-muted)]">Allocation</div>
                                    <div className="font-mono text-[var(--color-nexus-text-primary)]">{attribution_summary.allocation != null ? `${(attribution_summary.allocation * 100).toFixed(2)}%` : "--"}</div>
                                </div>
                                <div>
                                    <div className="text-[var(--color-nexus-text-muted)]">Selection</div>
                                    <div className="font-mono text-[var(--color-nexus-text-primary)]">{attribution_summary.selection != null ? `${(attribution_summary.selection * 100).toFixed(2)}%` : "--"}</div>
                                </div>
                                <div>
                                    <div className="text-[var(--color-nexus-text-muted)]">Interaction</div>
                                    <div className="font-mono text-[var(--color-nexus-text-primary)]">{attribution_summary.interaction != null ? `${(attribution_summary.interaction * 100).toFixed(2)}%` : "--"}</div>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">No attribution time series available.</div>
                    )}
                </article>
            </section>

            {attributionBars.length > 0 && (
                <section className="nexus-card p-6">
                    <h3 className="nexus-heading-3 font-semibold text-[var(--color-nexus-text-primary)] mb-4">Top Attribution Drivers</h3>
                    <div className="grid gap-3 md:grid-cols-2">
                        {attributionBars.map((row) => (
                            <article key={row.ticker} className="border border-[var(--color-nexus-border)] p-4 bg-[var(--color-nexus-surface-hover)]/25">
                                <div className="text-sm font-semibold text-[var(--color-nexus-text-primary)] mb-2">{row.ticker}</div>
                                <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                                    <div className="text-[var(--color-nexus-text-secondary)]">Alloc: {(row.allocation * 100).toFixed(2)}%</div>
                                    <div className="text-[var(--color-nexus-text-secondary)]">Select: {(row.selection * 100).toFixed(2)}%</div>
                                </div>
                            </article>
                        ))}
                    </div>
                </section>
            )}
        </div>
    );
}
