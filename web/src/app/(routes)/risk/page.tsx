"use client";

import { MetricCard } from "@/components/ui/MetricCard";
import { definitionTooltip } from "@/lib/definitions";
import { useNexus } from "@/components/nexus/NexusProvider";
import { EmptyState } from "@/components/nexus/EmptyState";
import { SkeletonCard, SkeletonBlock } from "@/components/nexus/Skeleton";
import { SectionContext } from "@/components/nexus/SectionContext";
import { BentoGrid, BentoItem } from "@/components/nexus/BentoGrid";
import { getMetricReasons, getMetricStatus } from "@/lib/coverageLogic";
import {
    Area,
    AreaChart,
    CartesianGrid,
    Cell,
    Line,
    LineChart,
    Pie,
    PieChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from "recharts";

export default function RiskPage() {
    const { state, status, error, mode, setMode, openRunCreator } = useNexus();

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
            <div className="space-y-8">
                <div>
                    <SkeletonBlock className="h-8 w-40 bg-[var(--color-nexus-surface)]" />
                    <SkeletonBlock className="mt-2 h-4 w-56 bg-[var(--color-nexus-surface)]" />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <SkeletonCard />
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
        risk,
        summary,
        performance,
        definitions,
        risk_contribution,
        rolling_metrics,
        manifest,
        coverage_summary,
        correlation_matrix,
    } = state;
    const currentDrawdown = performance.length > 0 ? performance[performance.length - 1]?.drawdown ?? null : null;
    const topRisk = risk_contribution?.contributions?.slice(0, 5) ?? [];
    const metricStatus = (kpiKey: string) => getMetricStatus(kpiKey, coverage_summary ?? null);
    const metricReasons = (kpiKey: string) => getMetricReasons(kpiKey, coverage_summary ?? null);
    const metricCoverage = (kpiKey: string) => (metricStatus(kpiKey) === "insufficient" ? "insufficient" : undefined);
    const metricReasonCodes = (kpiKey: string) =>
        metricStatus(kpiKey) === "insufficient" ? metricReasons(kpiKey) : undefined;
    const asOf = summary.last_date || manifest.timestamp;

    // Heatmap color logic (simple linear interpolation for red intensity)
    const getCorrelationColor = (value: number) => {
        if (value === 1) return "bg-[var(--color-nexus-primary)]/20 text-[var(--color-nexus-primary)]";
        if (value > 0.7) return "bg-[var(--color-nexus-danger)]/30 text-[var(--color-nexus-danger)]";
        if (value > 0.4) return "bg-[var(--color-nexus-danger)]/10 text-[var(--color-nexus-text-primary)]";
        return "bg-transparent text-[var(--color-nexus-text-secondary)]";
    };

    const COLORS = [
        "var(--color-nexus-primary)",
        "var(--color-nexus-secondary)",
        "var(--color-nexus-accent)",
        "var(--color-nexus-success)",
        "var(--color-nexus-warning)",
        "var(--color-nexus-danger)"
    ];

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            <div>
                <h2 className="text-3xl font-sans font-bold text-[var(--color-nexus-text-primary)] mb-2 tracking-tight">Risk Radar</h2>
                <p className="text-[var(--color-nexus-text-secondary)]">
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

            {summary.errors && summary.errors.length > 0 && (
                <div className="rounded-none border border-[var(--color-nexus-warning)] bg-[var(--color-nexus-warning)]/10 px-4 py-3 text-sm text-[var(--color-nexus-warning)] font-mono">
                    Some inputs are missing or incomplete for this run. Risk metrics may be limited.
                </div>
            )}

            <BentoGrid>
                {/* TOP ROW: KEY RISK METRICS - 4 Cards */}
                <BentoItem span={3} title="VaR (95%)">
                    <div className="flex flex-col justify-end h-full">
                        <span className="text-2xl font-mono font-bold text-[var(--color-nexus-text-primary)]">
                            {risk.var_95 !== null ? `${(risk.var_95 * 100).toFixed(2)}%` : "--"}
                        </span>
                        <span className="text-xs text-[var(--color-nexus-text-muted)] mt-1">Daily Value at Risk</span>
                    </div>
                </BentoItem>
                <BentoItem span={3} title="CVaR (95%)">
                    <div className="flex flex-col justify-end h-full">
                        <span className="text-2xl font-mono font-bold text-[var(--color-nexus-text-primary)]">
                            {risk.cvar_95 !== null ? `${(risk.cvar_95 * 100).toFixed(2)}%` : "--"}
                        </span>
                        <span className="text-xs text-[var(--color-nexus-text-muted)] mt-1">Conditional VaR</span>
                    </div>
                </BentoItem>
                <BentoItem span={3} title="Volatility">
                    <div className="flex flex-col justify-end h-full">
                        <span className="text-2xl font-mono font-bold text-[var(--color-nexus-text-primary)]">
                            {risk.volatility !== null ? `${(risk.volatility * 100).toFixed(2)}%` : "--"}
                        </span>
                        <span className="text-xs text-[var(--color-nexus-text-muted)] mt-1">Annualized (30d)</span>
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

                {/* HERO ROW: CORRELATION MATRIX (Span 8) + RISK CONTRIBUTION (Span 4) */}
                <BentoItem span={8} title="Correlation Matrix" rowSpan={2}>
                    {!correlation_matrix || correlation_matrix.status === "unavailable" ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">
                            Correlation matrix unavailable. {correlation_matrix?.reasons?.join(", ") || "Insufficient history."}
                        </div>
                    ) : (
                        <div className="overflow-x-auto h-full flex flex-col justify-center">
                            {correlation_matrix.reasons && correlation_matrix.reasons.length > 0 && (
                                <div className="text-xs text-[var(--color-nexus-text-muted)] mb-2">Notes: {correlation_matrix.reasons.join(", ")}</div>
                            )}
                            <table className="w-full border-collapse text-xs font-mono">
                                <thead>
                                    <tr>
                                        <th className="p-2 text-left font-semibold text-[var(--color-nexus-text-secondary)]"></th>
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
                                            <td className="p-2 font-semibold text-[var(--color-nexus-text-primary)] text-right pr-4">{row}</td>
                                            {correlation_matrix.assets_included.map((col) => {
                                                const val = correlation_matrix.matrix?.[row]?.[col];
                                                return (
                                                    <td key={`${row}-${col}`} className="p-1">
                                                        <div className={`w-full h-10 flex items-center justify-center rounded ${getCorrelationColor(val ?? 0)} transition-colors hover:ring-1 hover:ring-[var(--color-nexus-primary)]`}>
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
                </BentoItem>

                <BentoItem span={4} title="Risk Contribution" rowSpan={2}>
                    {topRisk.length === 0 ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">No risk contribution data available.</div>
                    ) : (
                        <div className="h-full flex flex-col">
                            <div className="flex-1 min-h-[200px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie
                                            data={topRisk}
                                            dataKey="volatility_pct"
                                            nameKey="ticker"
                                            cx="50%"
                                            cy="50%"
                                            innerRadius={60}
                                            outerRadius={80}
                                            paddingAngle={5}
                                            stroke="none"
                                        >
                                            {topRisk.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                            ))}
                                        </Pie>
                                        <Tooltip
                                            animationDuration={100}
                                            contentStyle={{
                                                backgroundColor: "var(--color-nexus-surface)",
                                                borderColor: "var(--color-nexus-border)",
                                                borderRadius: "4px",
                                                color: "var(--color-nexus-text-primary)",
                                                fontFamily: "var(--font-mono)",
                                                fontSize: "12px"
                                            }}
                                            formatter={(value) => [`${(Number(value) * 100).toFixed(1)}%`, "Volatility"]}
                                        />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                            <div className="mt-4 text-xs space-y-2">
                                {topRisk.slice(0, 5).map((item, index) => (
                                    <div key={item.ticker} className="flex items-center justify-between group">
                                        <div className="flex items-center gap-2">
                                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS[index % COLORS.length] }} />
                                            <span className="text-[var(--color-nexus-text-secondary)] group-hover:text-[var(--color-nexus-text-primary)] transition-colors">{item.ticker}</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <div className="w-16 h-1.5 bg-[var(--color-nexus-surface-hover)] rounded-full overflow-hidden">
                                                <div
                                                    className="h-full rounded-full"
                                                    style={{
                                                        width: `${item.volatility_pct * 100}%`,
                                                        backgroundColor: COLORS[index % COLORS.length]
                                                    }}
                                                />
                                            </div>
                                            <span className="font-mono text-[var(--color-nexus-text-primary)] w-8 text-right">{(item.volatility_pct * 100).toFixed(0)}%</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </BentoItem>

                {/* ROLLING RISK - Span 12 */}
                <BentoItem span={12} title="Rolling Volatility History" rowSpan={2}>
                    {!rolling_metrics || rolling_metrics.length === 0 ? (
                        <div className="text-sm text-[var(--color-nexus-text-secondary)]">No rolling risk metrics available.</div>
                    ) : (
                        <div className="flex-1 min-h-0 h-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={rolling_metrics}>
                                    <defs>
                                        <linearGradient id="colorVol" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="var(--color-nexus-warning)" stopOpacity={0.2} />
                                            <stop offset="95%" stopColor="var(--color-nexus-warning)" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
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
                                        labelStyle={{ color: "var(--color-nexus-text-secondary)", marginBottom: "0.5rem" }}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="rolling_volatility"
                                        stroke="var(--color-nexus-warning)"
                                        strokeWidth={2}
                                        fill="url(#colorVol)"
                                        name="Rolling Volatility"
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </BentoItem>
            </BentoGrid>
        </div>
    );
}
