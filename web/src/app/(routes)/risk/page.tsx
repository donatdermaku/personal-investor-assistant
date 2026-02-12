"use client";

import { useNexus } from "@/components/nexus/NexusProvider";
import { EmptyState } from "@/components/nexus/EmptyState";
import { SkeletonBlock } from "@/components/nexus/Skeleton";
import { SectionContext } from "@/components/nexus/SectionContext";
import { LazyChart } from "@/components/nexus/LazyChart";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { downsampleSeries } from "@/lib/chartPerformance";
import { Area, AreaChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const COLORS = [
    "var(--color-nexus-primary)",
    "var(--color-nexus-secondary)",
    "var(--color-nexus-accent)",
    "var(--color-nexus-success)",
    "var(--color-nexus-warning)",
    "var(--color-nexus-danger)",
];

export default function RiskPage() {
    const { state, status, error, mode, setMode, openRunCreator } = useNexus();
    const isPhone = useMediaQuery("(max-width: 767px)");

    if (status === "error") {
        return (
            <EmptyState
                title="Unable to load risk data"
                description={error || "Check API connectivity, then retry in Live Mode or continue in Demo Mode."}
                primaryAction={{ label: "Retry in Live Mode", onClick: () => setMode("live") }}
                secondaryAction={{ label: "Switch to Demo Mode", onClick: () => setMode("demo") }}
            />
        );
    }

    if (status === "empty") {
        return (
            <EmptyState
                title="No risk metrics yet"
                description="Generate a run to see VaR, drawdowns, and concentration risk."
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
                <div className="nexus-metrics-grid metrics-4">
                    <SkeletonBlock className="h-28 bg-[var(--color-nexus-surface)]" />
                    <SkeletonBlock className="h-28 bg-[var(--color-nexus-surface)]" />
                    <SkeletonBlock className="h-28 bg-[var(--color-nexus-surface)]" />
                    <SkeletonBlock className="h-28 bg-[var(--color-nexus-surface)]" />
                </div>
            </div>
        );
    }

    const { risk, summary, rolling_metrics, risk_contribution, correlation_matrix } = state;
    const topRisk = risk_contribution?.contributions?.slice(0, 6) ?? [];
    const lightRolling = downsampleSeries(rolling_metrics ?? [], isPhone ? 120 : 260);
    const getCorrelationColor = (value: number) => {
        if (value === 1) return "bg-[var(--color-nexus-primary)]/20 text-[var(--color-nexus-primary)]";
        if (value > 0.7) return "bg-[var(--color-nexus-danger)]/30 text-[var(--color-nexus-danger)]";
        if (value > 0.4) return "bg-[var(--color-nexus-danger)]/10 text-[var(--color-nexus-text-primary)]";
        return "bg-transparent text-[var(--color-nexus-text-secondary)]";
    };

    return (
        <div className="nexus-page animate-in fade-in duration-500">
            <div>
                <h2 className="nexus-heading-1 font-sans font-bold text-[var(--color-nexus-text-primary)] mb-2 tracking-tight">Risk Radar</h2>
                <p className="text-[var(--color-nexus-text-secondary)] text-base">
                    {mode === "demo" ? "Demo risk metrics and analysis" : "Portfolio risk metrics and analysis"}
                </p>
            </div>

            <SectionContext
                title="Risk Context"
                items={[
                    {
                        label: "What it measures",
                        text: "Downside risk, volatility, and drawdown behavior based on the portfolio return history.",
                    },
                    {
                        label: "Why it matters",
                        text: "Explains the variability and tail exposure behind the return profile.",
                    },
                ]}
            />

            <section className="nexus-metrics-grid metrics-4">
                <article className="nexus-card p-6">
                    <div className="text-label mb-3">VaR (95%)</div>
                    <div className="nexus-number font-mono font-bold text-[var(--color-nexus-text-primary)]">
                        {risk.var_95 !== null ? `${(risk.var_95 * 100).toFixed(2)}%` : "--"}
                    </div>
                </article>
                <article className="nexus-card p-6">
                    <div className="text-label mb-3">CVaR (95%)</div>
                    <div className="nexus-number font-mono font-bold text-[var(--color-nexus-text-primary)]">
                        {risk.cvar_95 !== null ? `${(risk.cvar_95 * 100).toFixed(2)}%` : "--"}
                    </div>
                </article>
                <article className="nexus-card p-6">
                    <div className="text-label mb-3">Volatility</div>
                    <div className="nexus-number font-mono font-bold text-[var(--color-nexus-text-primary)]">
                        {risk.volatility !== null ? `${(risk.volatility * 100).toFixed(2)}%` : "--"}
                    </div>
                </article>
                <article className="nexus-card p-6">
                    <div className="text-label mb-3">Max Drawdown</div>
                    <div className="nexus-number font-mono font-bold text-[var(--color-nexus-danger)]">
                        {summary.max_drawdown !== null ? `${(summary.max_drawdown * 100).toFixed(2)}%` : "--"}
                    </div>
                </article>
            </section>

            <section className="nexus-card p-6">
                <h3 className="nexus-heading-3 font-semibold text-[var(--color-nexus-text-primary)] mb-4">Correlation Matrix</h3>
                {!correlation_matrix || correlation_matrix.status === "unavailable" ? (
                    <div className="text-sm text-[var(--color-nexus-text-secondary)]">
                        Correlation matrix unavailable. {correlation_matrix?.reasons?.join(", ") || "Insufficient history."}
                    </div>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full min-w-[640px] border-collapse text-xs font-mono">
                            <thead>
                                <tr>
                                    <th className="p-2 text-left font-semibold text-[var(--color-nexus-text-secondary)]" />
                                    {correlation_matrix.assets_included.map((ticker) => (
                                        <th key={ticker} className="p-2 text-center font-semibold text-[var(--color-nexus-text-secondary)]">
                                            {ticker}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {correlation_matrix.assets_included.map((row) => (
                                    <tr key={row}>
                                        <td className="p-2 font-semibold text-[var(--color-nexus-text-primary)] text-right pr-3">{row}</td>
                                        {correlation_matrix.assets_included.map((col) => {
                                            const val = correlation_matrix.matrix?.[row]?.[col];
                                            return (
                                                <td key={`${row}-${col}`} className="p-1">
                                                    <div className={`w-full h-10 flex items-center justify-center ${getCorrelationColor(val ?? 0)}`}>
                                                        {val?.toFixed(2) ?? "--"}
                                                    </div>
                                                </td>
                                            );
                                        })}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </section>

            <section className="grid gap-[var(--grid-gap)] min-[1200px]:grid-cols-3">
                <article className="nexus-card p-6 min-h-[320px] min-[1200px]:col-span-2">
                    <h3 className="nexus-heading-3 font-semibold text-[var(--color-nexus-text-primary)] mb-4">Rolling Volatility</h3>
                    {!rolling_metrics || rolling_metrics.length === 0 ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">No rolling risk metrics available.</div>
                    ) : (
                        <LazyChart heightClassName="h-[260px] min-[1200px]:h-[320px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={lightRolling}>
                                    <defs>
                                        <linearGradient id="rollingVol" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="var(--color-nexus-warning)" stopOpacity={0.2} />
                                            <stop offset="95%" stopColor="var(--color-nexus-warning)" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-nexus-border)" vertical={false} />
                                    <XAxis dataKey="date" hide />
                                    <YAxis tickLine={false} axisLine={false} tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`} />
                                    <Tooltip formatter={(value) => (value != null ? [`${(Number(value) * 100).toFixed(2)}%`, "Volatility"] : ["--", "Volatility"])} />
                                    <Area type="monotone" dataKey="rolling_volatility" stroke="var(--color-nexus-warning)" strokeWidth={2} fill="url(#rollingVol)" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </LazyChart>
                    )}
                </article>

                <article className="nexus-card p-6 min-h-[320px]">
                    <h3 className="nexus-heading-3 font-semibold text-[var(--color-nexus-text-primary)] mb-4">Risk Contribution</h3>
                    {topRisk.length === 0 ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">No risk contribution data available.</div>
                    ) : (
                        <div className="space-y-4">
                            <LazyChart heightClassName="h-[180px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie data={topRisk} dataKey="volatility_pct" nameKey="ticker" cx="50%" cy="50%" innerRadius={45} outerRadius={72}>
                                            {topRisk.map((_, index) => (
                                                <Cell key={`risk-contrib-${index}`} fill={COLORS[index % COLORS.length]} />
                                            ))}
                                        </Pie>
                                        <Tooltip formatter={(value) => [`${(Number(value) * 100).toFixed(1)}%`, "Volatility"]} />
                                    </PieChart>
                                </ResponsiveContainer>
                            </LazyChart>
                            <div className="space-y-2">
                                {topRisk.map((item, index) => (
                                    <div key={item.ticker} className="flex items-center justify-between text-xs">
                                        <div className="flex items-center gap-2">
                                            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS[index % COLORS.length] }} />
                                            <span className="text-[var(--color-nexus-text-secondary)]">{item.ticker}</span>
                                        </div>
                                        <span className="font-mono text-[var(--color-nexus-text-primary)]">{(item.volatility_pct * 100).toFixed(1)}%</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </article>
            </section>
        </div>
    );
}
