"use client";

import { useEffect, useState } from "react";
import { MetricCard } from "@/components/ui/MetricCard";
import { getNexusState } from "@/lib/api";
import { definitionTooltip } from "@/lib/definitions";
import { NexusState } from "@/types/nexus";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function OverviewPage() {
    const [state, setState] = useState<NexusState | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        getNexusState().then(data => {
            setState(data);
            setLoading(false);
        });
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-[400px]">
                <div className="text-gray-500">Loading portfolio data...</div>
            </div>
        );
    }

    if (!state) {
        return (
            <div className="flex items-center justify-center min-h-[400px]">
                <div className="text-red-500">Failed to load data</div>
            </div>
        );
    }

    const { summary, equity_curve, definitions } = state;

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold text-[#0F172A] mb-2">Overview</h2>
                <p className="text-gray-600">Portfolio performance summary</p>
            </div>

            {/* KPI Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <MetricCard
                    label="Strategy Return (TWR)"
                    value={summary.twr !== null ? `${(summary.twr * 100).toFixed(2)}%` : "--"}
                    tooltip={definitionTooltip(definitions, "twr")}
                />
                <MetricCard
                    label="Personal Return (MWR)"
                    value={summary.mwr !== null ? `${(summary.mwr * 100).toFixed(2)}%` : "--"}
                    tooltip={definitionTooltip(definitions, "mwr")}
                />
                <MetricCard
                    label="Portfolio Value"
                    value={summary.final_value !== null ? `$${summary.final_value.toLocaleString()}` : "--"}
                    subtext={summary.last_date || undefined}
                />
                <MetricCard
                    label="Max Drawdown"
                    value={summary.max_drawdown !== null ? `${(summary.max_drawdown * 100).toFixed(2)}%` : "--"}
                    tooltip={definitionTooltip(definitions, "max_drawdown")}
                />
            </div>

            {/* Equity Curve */}
            <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-lg font-semibold text-[#0F172A] mb-4">Equity Curve</h3>
                {equity_curve.length === 0 ? (
                    <div className="text-sm text-gray-500">No equity curve data available.</div>
                ) : (
                    <ResponsiveContainer width="100%" height={400}>
                        <LineChart data={equity_curve}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                            <XAxis
                                dataKey="date"
                                stroke="#6B7280"
                                style={{ fontSize: '12px' }}
                            />
                            <YAxis
                                stroke="#6B7280"
                                style={{ fontSize: '12px' }}
                                tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'white',
                                    border: '1px solid #E5E7EB',
                                    borderRadius: '0.5rem'
                                }}
                                formatter={(value) => value != null ? [`$${Number(value).toLocaleString()}`, 'Value'] : ['--', 'Value']}
                            />
                            <Legend />
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

            {/* Errors */}
            {summary.errors && summary.errors.length > 0 && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <h4 className="font-semibold text-red-900 mb-2">⚠️ Portfolio Errors</h4>
                    <ul className="list-disc list-inside text-red-700 text-sm space-y-1">
                        {summary.errors.map((err, idx) => (
                            <li key={idx}>{err}</li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}
