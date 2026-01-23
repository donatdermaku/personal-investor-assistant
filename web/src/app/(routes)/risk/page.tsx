"use client";

import { useEffect, useState } from "react";
import { MetricCard } from "@/components/ui/MetricCard";
import { getNexusState } from "@/lib/api";
import { definitionTooltip } from "@/lib/definitions";
import { NexusState } from "@/types/nexus";

export default function RiskPage() {
    const [state, setState] = useState<NexusState | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        getNexusState().then(data => {
            setState(data);
            setLoading(false);
        });
    }, []);

    if (loading) {
        return <div className="text-gray-500">Loading...</div>;
    }

    if (!state) {
        return <div className="text-red-500">Failed to load data</div>;
    }

    const { risk, summary, performance, definitions } = state;
    const currentDrawdown = performance.length > 0 ? performance[performance.length - 1]?.drawdown ?? null : null;

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold text-[#0F172A] mb-2">Risk</h2>
                <p className="text-gray-600">Portfolio risk metrics and analysis</p>
            </div>

            {/* Risk Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <MetricCard
                    label="VaR (95%)"
                    value={risk.var_95 !== null ? `${(risk.var_95 * 100).toFixed(2)}%` : "--"}
                    tooltip={definitionTooltip(definitions, "var_daily")}
                    subtext="Daily"
                />
                <MetricCard
                    label="CVaR (95%)"
                    value={risk.cvar_95 !== null ? `${(risk.cvar_95 * 100).toFixed(2)}%` : "--"}
                    tooltip={definitionTooltip(definitions, "cvar_daily")}
                    subtext="Daily"
                />
                <MetricCard
                    label="Volatility"
                    value={risk.volatility !== null ? `${(risk.volatility * 100).toFixed(2)}%` : "--"}
                    tooltip={definitionTooltip(definitions, "rolling_volatility")}
                />
                <MetricCard
                    label="Sharpe Ratio"
                    value={risk.sharpe !== null ? risk.sharpe.toFixed(2) : "--"}
                    tooltip={definitionTooltip(definitions, "sharpe_rolling")}
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

            {/* Correlation Matrix Placeholder */}
            <div className="bg-white border border-gray-200 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#0F172A] mb-4">Correlation Matrix</h3>
                <p className="text-gray-500 text-sm">
                    Holdings correlation heatmap will be implemented when backend provides correlation data.
                </p>
            </div>
        </div>
    );
}
