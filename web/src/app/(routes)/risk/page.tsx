"use client";

import { MetricCard } from "@/components/ui/MetricCard";
import { definitionTooltip } from "@/lib/definitions";
import { useNexus } from "@/components/nexus/NexusProvider";
import { EmptyState } from "@/components/nexus/EmptyState";
import { SkeletonCard, SkeletonBlock } from "@/components/nexus/Skeleton";
import { SectionContext } from "@/components/nexus/SectionContext";
import { coverageStatus } from "@/lib/coverage";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export default function RiskPage() {
    const { state, status, error, mode, setMode, openRunCreator } = useNexus();

    if (status === "error") {
        return (
            <EmptyState
                title="Unable to load risk data"
                description={error || "Check your backend connection and try again."}
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
                    <SkeletonBlock className="h-8 w-40" />
                    <SkeletonBlock className="mt-2 h-4 w-56" />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <SkeletonCard />
                    <SkeletonCard />
                    <SkeletonCard />
                    <SkeletonCard />
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-6">
                    <SkeletonBlock className="h-5 w-32" />
                    <SkeletonBlock className="mt-4 h-28 w-full" />
                </div>
            </div>
        );
    }

    const { risk, summary, performance, definitions, risk_contribution, rolling_metrics, manifest } = state;
    const currentDrawdown = performance.length > 0 ? performance[performance.length - 1]?.drawdown ?? null : null;
    const topRisk = risk_contribution?.contributions?.slice(0, 6) ?? [];
    const coverage = coverageStatus(manifest);
    const asOf = summary.last_date || manifest.timestamp;

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold text-[#0F172A] mb-2">Risk</h2>
                <p className="text-gray-600">
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
                    {
                        label: "When it misleads",
                        text: "Short histories and cash-only positions can make risk statistics appear muted.",
                    },
                    {
                        label: "Assumptions",
                        text: "Risk metrics use daily returns and require sufficient history for stability.",
                    },
                ]}
            />

            {summary.errors && summary.errors.length > 0 && (
                <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                    Some inputs are missing or incomplete for this run. Risk metrics may be limited.
                </div>
            )}

            {/* Risk Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <MetricCard
                    label="VaR (95%)"
                    value={risk.var_95 !== null ? `${(risk.var_95 * 100).toFixed(2)}%` : "--"}
                    tooltip={definitionTooltip(definitions, "var_daily")}
                    subtext="Daily"
                    coverageStatus={risk.var_95 !== null ? coverage : "insufficient"}
                    asOf={asOf}
                />
                <MetricCard
                    label="CVaR (95%)"
                    value={risk.cvar_95 !== null ? `${(risk.cvar_95 * 100).toFixed(2)}%` : "--"}
                    tooltip={definitionTooltip(definitions, "cvar_daily")}
                    subtext="Daily"
                    coverageStatus={risk.cvar_95 !== null ? coverage : "insufficient"}
                    asOf={asOf}
                />
                <MetricCard
                    label="Volatility"
                    value={risk.volatility !== null ? `${(risk.volatility * 100).toFixed(2)}%` : "--"}
                    tooltip={definitionTooltip(definitions, "rolling_volatility")}
                    coverageStatus={risk.volatility !== null ? coverage : "insufficient"}
                    asOf={asOf}
                />
                <MetricCard
                    label="Sharpe Ratio"
                    value={risk.sharpe !== null ? risk.sharpe.toFixed(2) : "--"}
                    tooltip={definitionTooltip(definitions, "sharpe_rolling")}
                    coverageStatus={risk.sharpe !== null ? coverage : "insufficient"}
                    asOf={asOf}
                />
            </div>

            {/* Drawdown */}
            <div className="bg-white border border-gray-200 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#0F172A] mb-4">Drawdown Analysis</h3>
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <div className="text-sm text-gray-500 mb-1">Max Drawdown</div>
                        <div className="text-2xl font-bold text-[#0F172A]">
                            {summary.max_drawdown !== null ? `${(summary.max_drawdown * 100).toFixed(2)}%` : "--"}
                        </div>
                    </div>
                    <div>
                        <div className="text-sm text-gray-500 mb-1">Current Drawdown</div>
                        <div className="text-2xl font-bold text-[#0F172A]">
                            {currentDrawdown !== null && currentDrawdown !== undefined
                                ? `${(currentDrawdown * 100).toFixed(2)}%`
                                : "--"}
                        </div>
                        <div className="text-xs text-gray-400 mt-1">Latest observation</div>
                    </div>
                </div>
            </div>

            <details className="bg-white border border-gray-200 rounded-lg p-6">
                <summary className="cursor-pointer text-lg font-semibold text-[#0F172A] mb-4">Correlation Matrix</summary>
                <p className="text-gray-500 text-sm">
                    Correlation data is unavailable in this release. This section appears once backend coverage includes correlation outputs.
                </p>
            </details>

            <details className="bg-white border border-gray-200 rounded-lg p-6">
                <summary className="cursor-pointer text-lg font-semibold text-[#0F172A] mb-4">Risk Contribution</summary>
                {topRisk.length === 0 ? (
                    <div className="text-sm text-gray-500">No risk contribution data available.</div>
                ) : (
                    <div className="divide-y divide-gray-200 text-sm">
                        {topRisk.map((row) => (
                            <div key={row.ticker} className="py-2 flex items-center justify-between">
                                <span className="font-medium text-gray-700">{row.ticker}</span>
                                <span className="text-gray-500">
                                    {(row.volatility_pct * 100).toFixed(1)}% of volatility
                                </span>
                            </div>
                        ))}
                    </div>
                )}
            </details>

            <details className="bg-white border border-gray-200 rounded-lg p-6">
                <summary className="cursor-pointer text-lg font-semibold text-[#0F172A] mb-4">Rolling Risk</summary>
                {!rolling_metrics || rolling_metrics.length === 0 ? (
                    <div className="text-sm text-gray-500">No rolling risk metrics available.</div>
                ) : (
                    <ResponsiveContainer width="100%" height={240}>
                        <LineChart data={rolling_metrics}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#E3E7EE" />
                            <XAxis dataKey="date" stroke="#64748B" style={{ fontSize: "12px" }} />
                            <YAxis
                                yAxisId="left"
                                stroke="#64748B"
                                style={{ fontSize: "12px" }}
                                tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`}
                            />
                            <YAxis
                                yAxisId="right"
                                orientation="right"
                                stroke="#94A3B8"
                                style={{ fontSize: "12px" }}
                                tickFormatter={(value) => Number(value).toFixed(1)}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: "white",
                                    border: "1px solid #E3E7EE",
                                    borderRadius: "0.5rem",
                                }}
                                formatter={(value, name) => {
                                    if (name === "Rolling Sharpe") {
                                        return value != null ? [Number(value).toFixed(2), name] : ["--", name];
                                    }
                                    return value != null ? [`${(Number(value) * 100).toFixed(2)}%`, name] : ["--", name];
                                }}
                            />
                            <Line
                                yAxisId="left"
                                type="monotone"
                                dataKey="rolling_volatility"
                                stroke="#2563EB"
                                strokeWidth={2}
                                dot={false}
                                name="Rolling Volatility"
                            />
                            <Line
                                yAxisId="right"
                                type="monotone"
                                dataKey="rolling_sharpe"
                                stroke="#16A34A"
                                strokeWidth={2}
                                dot={false}
                                name="Rolling Sharpe"
                            />
                            <Line
                                yAxisId="left"
                                type="monotone"
                                dataKey="rolling_drawdown"
                                stroke="#F97316"
                                strokeWidth={2}
                                dot={false}
                                name="Rolling Drawdown"
                            />
                        </LineChart>
                    </ResponsiveContainer>
                )}
            </details>
        </div>
    );
}
