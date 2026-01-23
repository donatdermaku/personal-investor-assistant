"use client";

import { useEffect, useState } from "react";
import { MetricCard } from "@/components/ui/MetricCard";
import { getNexusState } from "@/lib/api";
import { definitionTooltip } from "@/lib/definitions";
import { NexusState } from "@/types/nexus";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts";

export default function PerformancePage() {
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

    const { summary, performance, monthly_returns, definitions } = state;
    const firstValue = performance[0]?.value ?? null;
    const lastValue = performance[performance.length - 1]?.value ?? null;
    const totalReturn =
        firstValue && lastValue ? (lastValue / firstValue) - 1 : null;

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold text-[#0F172A] mb-2">Performance</h2>
                <p className="text-gray-600">Detailed returns analysis</p>
            </div>

            {/* Returns Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <MetricCard
                    label="TWR (Strategy)"
                    value={summary.twr !== null ? `${(summary.twr * 100).toFixed(2)}%` : "--"}
                    tooltip={definitionTooltip(definitions, "twr")}
                />
                <MetricCard
                    label="MWR (Personal)"
                    value={summary.mwr !== null ? `${(summary.mwr * 100).toFixed(2)}%` : "--"}
                    tooltip={definitionTooltip(definitions, "mwr")}
                />
                <MetricCard
                    label="Total Return"
                    value={totalReturn !== null ? `${(totalReturn * 100).toFixed(2)}%` : "--"}
                    subtext="From first valuation"
                />
            </div>

            {/* Portfolio Value */}
            <div className="bg-white border border-gray-200 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#0F172A] mb-4">Portfolio Value</h3>
                {performance.length === 0 ? (
                    <div className="text-sm text-gray-500">No performance data available.</div>
                ) : (
                    <ResponsiveContainer width="100%" height={320}>
                        <LineChart data={performance}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                            <XAxis dataKey="date" stroke="#6B7280" style={{ fontSize: "12px" }} />
                            <YAxis stroke="#6B7280" style={{ fontSize: "12px" }} />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: "white",
                                    border: "1px solid #E5E7EB",
                                    borderRadius: "0.5rem",
                                }}
                                formatter={(value) =>
                                    value != null ? [`$${Number(value).toLocaleString()}`, "Value"] : ["--", "Value"]
                                }
                            />
                            <Line
                                type="monotone"
                                dataKey="value"
                                stroke="#0F172A"
                                strokeWidth={2}
                                dot={false}
                                name="Portfolio Value"
                            />
                        </LineChart>
                    </ResponsiveContainer>
                )}
            </div>

            {/* Drawdowns */}
            <div className="bg-white border border-gray-200 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#0F172A] mb-4">Drawdowns</h3>
                {performance.length === 0 ? (
                    <div className="text-sm text-gray-500">No drawdown data available.</div>
                ) : (
                    <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={performance}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                            <XAxis dataKey="date" stroke="#6B7280" style={{ fontSize: "12px" }} />
                            <YAxis
                                stroke="#6B7280"
                                style={{ fontSize: "12px" }}
                                tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: "white",
                                    border: "1px solid #E5E7EB",
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
            </div>

            {/* Monthly Returns */}
            <div className="bg-white border border-gray-200 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#0F172A] mb-4">Monthly Returns</h3>
                {monthly_returns.length === 0 ? (
                    <div className="text-sm text-gray-500">No monthly return data available.</div>
                ) : (
                    <ResponsiveContainer width="100%" height={260}>
                        <BarChart data={monthly_returns}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                            <XAxis dataKey="date" stroke="#6B7280" style={{ fontSize: "12px" }} />
                            <YAxis
                                stroke="#6B7280"
                                style={{ fontSize: "12px" }}
                                tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: "white",
                                    border: "1px solid #E5E7EB",
                                    borderRadius: "0.5rem",
                                }}
                                formatter={(value) =>
                                    value != null ? [`${(Number(value) * 100).toFixed(2)}%`, "Return"] : ["--", "Return"]
                                }
                            />
                            <Bar dataKey="return" fill="#0F172A" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                )}
            </div>

            {/* Attribution */}
            <div className="bg-white border border-gray-200 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#0F172A] mb-2">Attribution</h3>
                <p className="text-gray-500 text-sm">
                    Attribution by holding and sector will appear once backend contribution data is exposed.
                </p>
            </div>

            {/* Factor Tilts */}
            <div className="bg-white border border-gray-200 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#0F172A] mb-2">Factor Tilts</h3>
                <p className="text-gray-500 text-sm">
                    Factor tilt data will populate when the API provides portfolio factor distributions.
                </p>
            </div>
        </div>
    );
}
