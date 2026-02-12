"use client";

import { useNexus } from "@/components/nexus/NexusProvider";
import { EmptyState } from "@/components/nexus/EmptyState";
import { SkeletonBlock } from "@/components/nexus/Skeleton";
import { SectionContext } from "@/components/nexus/SectionContext";
import { LazyChart } from "@/components/nexus/LazyChart";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { downsampleSeries } from "@/lib/chartPerformance";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function OverviewPage() {
    const { state, status, error, mode, setMode, openRunCreator } = useNexus();
    const isPhone = useMediaQuery("(max-width: 767px)");

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
            <div className="nexus-page">
                <div>
                    <SkeletonBlock className="h-8 w-48 bg-[var(--color-nexus-surface)]" />
                    <SkeletonBlock className="mt-2 h-4 w-64 bg-[var(--color-nexus-surface)]" />
                </div>
                <div className="nexus-metrics-grid metrics-4">
                    <SkeletonBlock className="h-28 bg-[var(--color-nexus-surface)]" />
                    <SkeletonBlock className="h-28 bg-[var(--color-nexus-surface)]" />
                    <SkeletonBlock className="h-28 bg-[var(--color-nexus-surface)]" />
                    <SkeletonBlock className="h-28 bg-[var(--color-nexus-surface)]" />
                </div>
            </div>
        );
    }

    const { summary, performance, holdings, risk, diagnostics } = state;
    const lastValue = performance[performance.length - 1]?.value ?? null;
    const topHoldings = holdings.slice(0, 8);
    const sampledPerformance = downsampleSeries(performance, isPhone ? 120 : 280);

    return (
        <div className="nexus-page animate-in fade-in duration-500">
            <div>
                <h2 className="nexus-heading-1 font-sans font-bold text-[var(--color-nexus-text-primary)] mb-2 tracking-tight">Overview</h2>
                <p className="text-[var(--color-nexus-text-secondary)] text-base">
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

            <section className="nexus-metrics-grid metrics-4">
                <article className="nexus-card p-6">
                    <div className="text-label mb-3">TWR (YTD)</div>
                    <div className={`metric-value-responsive nexus-number font-mono font-bold ${summary.twr && summary.twr >= 0 ? "text-[var(--color-nexus-success)]" : "text-[var(--color-nexus-danger)]"}`}>
                        {summary.twr !== null ? `${(summary.twr * 100).toFixed(2)}%` : "--"}
                    </div>
                </article>
                <article className="nexus-card p-6">
                    <div className="text-label mb-3">MWR (YTD)</div>
                    <div className={`metric-value-responsive nexus-number font-mono font-bold ${summary.mwr && summary.mwr >= 0 ? "text-[var(--color-nexus-success)]" : "text-[var(--color-nexus-danger)]"}`}>
                        {summary.mwr !== null ? `${(summary.mwr * 100).toFixed(2)}%` : "--"}
                    </div>
                </article>
                <article className="nexus-card p-6">
                    <div className="text-label mb-3">Portfolio Value</div>
                    <div className="metric-value-responsive nexus-number font-mono font-bold text-[var(--color-nexus-text-primary)]">
                        {lastValue !== null ? `$${lastValue.toLocaleString()}` : "--"}
                    </div>
                </article>
                <article className="nexus-card p-6">
                    <div className="text-label mb-3">Max Drawdown</div>
                    <div className="metric-value-responsive nexus-number font-mono font-bold text-[var(--color-nexus-danger)]">
                        {summary.max_drawdown !== null ? `${(summary.max_drawdown * 100).toFixed(2)}%` : "--"}
                    </div>
                </article>
            </section>

            <section className="grid gap-[var(--grid-gap)] min-[1200px]:grid-cols-3">
                <article className="nexus-card p-6 min-h-[340px] min-[1200px]:col-span-2">
                    <h3 className="nexus-heading-3 font-semibold text-[var(--color-nexus-text-primary)] mb-4">Equity Curve</h3>
                    {performance.length === 0 ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">No equity curve data available.</div>
                    ) : (
                        <LazyChart heightClassName="h-[300px] min-[1200px]:h-[360px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={sampledPerformance}>
                                    <defs>
                                        <linearGradient id="overviewEquity" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="var(--color-nexus-primary)" stopOpacity={0.25} />
                                            <stop offset="95%" stopColor="var(--color-nexus-primary)" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-nexus-border)" vertical={false} />
                                    <XAxis dataKey="date" tickLine={false} axisLine={false} minTickGap={24} stroke="var(--color-nexus-text-muted)" />
                                    <YAxis tickLine={false} axisLine={false} stroke="var(--color-nexus-text-muted)" />
                                    <Tooltip
                                        animationDuration={100}
                                        contentStyle={{
                                            backgroundColor: "var(--color-nexus-surface)",
                                            borderColor: "var(--color-nexus-primary)",
                                            borderRadius: "0",
                                            color: "var(--color-nexus-text-primary)",
                                        }}
                                        formatter={(value) => (value != null ? [`$${Number(value).toLocaleString()}`, "Value"] : ["--", "Value"])}
                                    />
                                    <Area type="monotone" dataKey="value" stroke="var(--color-nexus-primary)" strokeWidth={2} fill="url(#overviewEquity)" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </LazyChart>
                    )}
                </article>

                <article className="nexus-card p-6 min-h-[340px]">
                    <h3 className="nexus-heading-3 font-semibold text-[var(--color-nexus-text-primary)] mb-4">Stats</h3>
                    <div className="space-y-3">
                        <div className="flex items-center justify-between border-b border-[var(--color-nexus-border)] pb-2">
                            <span className="text-sm text-[var(--color-nexus-text-secondary)]">Volatility (30d)</span>
                            <span className="text-sm font-mono text-[var(--color-nexus-text-primary)]">{risk.volatility != null ? `${(risk.volatility * 100).toFixed(2)}%` : "--"}</span>
                        </div>
                        <div className="flex items-center justify-between border-b border-[var(--color-nexus-border)] pb-2">
                            <span className="text-sm text-[var(--color-nexus-text-secondary)]">Sharpe Ratio</span>
                            <span className="text-sm font-mono text-[var(--color-nexus-text-primary)]">{risk.sharpe != null ? risk.sharpe.toFixed(2) : "--"}</span>
                        </div>
                        <div className="flex items-center justify-between border-b border-[var(--color-nexus-border)] pb-2">
                            <span className="text-sm text-[var(--color-nexus-text-secondary)]">Beta to SPY</span>
                            <span className="text-sm font-mono text-[var(--color-nexus-text-primary)]">{risk.beta != null ? risk.beta.toFixed(2) : "--"}</span>
                        </div>
                    </div>
                </article>
            </section>

            <section className="nexus-card overflow-hidden">
                <div className="p-6 border-b border-[var(--color-nexus-border)]">
                    <h3 className="nexus-heading-3 font-semibold text-[var(--color-nexus-text-primary)]">Holdings Snapshot</h3>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full min-w-[680px] text-sm">
                        <thead className="bg-[var(--color-nexus-surface-hover)] sticky top-0">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs text-[var(--color-nexus-text-secondary)] uppercase tracking-wider">Ticker</th>
                                <th className="px-6 py-3 text-right text-xs text-[var(--color-nexus-text-secondary)] uppercase tracking-wider">Shares</th>
                                <th className="px-6 py-3 text-right text-xs text-[var(--color-nexus-text-secondary)] uppercase tracking-wider">Weight</th>
                                <th className="px-6 py-3 text-right text-xs text-[var(--color-nexus-text-secondary)] uppercase tracking-wider">Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {topHoldings.map((holding, idx) => (
                                <tr key={`${holding.ticker}-${idx}`} className="border-t border-[var(--color-nexus-border)]">
                                    <td className="px-6 py-4 font-mono text-[var(--color-nexus-text-primary)]">{holding.ticker}</td>
                                    <td className="px-6 py-4 text-right font-mono text-[var(--color-nexus-text-secondary)]">{holding.shares != null ? holding.shares.toLocaleString() : "--"}</td>
                                    <td className="px-6 py-4 text-right font-mono text-[var(--color-nexus-text-secondary)]">{holding.weight != null ? `${(holding.weight * 100).toFixed(2)}%` : "--"}</td>
                                    <td className="px-6 py-4 text-right font-mono text-[var(--color-nexus-text-primary)]">{holding.value != null ? `$${holding.value.toLocaleString()}` : "--"}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </section>

            <section className="nexus-card p-6">
                <h3 className="nexus-heading-3 font-semibold text-[var(--color-nexus-text-primary)] mb-4">System Diagnostics</h3>
                {!diagnostics || diagnostics.length === 0 ? (
                    <div className="text-sm text-[var(--color-nexus-success)]">All systems nominal.</div>
                ) : (
                    <div className="grid gap-3 md:grid-cols-2">
                        {diagnostics.slice(0, 6).map((diag, idx) => (
                            <article key={`${diag.key}-${idx}`} className="border border-[var(--color-nexus-border)] bg-[var(--color-nexus-surface-hover)]/30 p-4">
                                <div className="text-xs uppercase tracking-wider text-[var(--color-nexus-text-secondary)] mb-1">{diag.category}</div>
                                <div className="text-sm font-semibold text-[var(--color-nexus-text-primary)]">{diag.check || diag.summary}</div>
                                <div className="text-xs text-[var(--color-nexus-text-muted)] mt-1">{diag.message || diag.summary}</div>
                            </article>
                        ))}
                    </div>
                )}
            </section>
        </div>
    );
}
