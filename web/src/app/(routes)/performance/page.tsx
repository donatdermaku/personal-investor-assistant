"use client";

import { useEffect, useState } from "react";
import { MetricCard } from "@/components/ui/MetricCard";
import { getNexusState } from "@/lib/api";
import { NexusState } from "@/types/nexus";

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

    const { summary } = state;

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
                    tooltip="Time-Weighted Return"
                />
                <MetricCard
                    label="MWR (Personal)"
                    value={summary.mwr !== null ? `${(summary.mwr * 100).toFixed(2)}%` : "--"}
                    tooltip="Money-Weighted Return (IRR)"
                />
                <MetricCard
                    label="Total Return"
                    value={summary.final_value && summary.final_value > 0
                        ? `${((summary.final_value / 100000 - 1) * 100).toFixed(2)}%`
                        : "--"}
                    subtext="vs. initial $100k"
                />
            </div>

            {/* Attribution Placeholder */}
            <div className="bg-white border border-gray-200 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#0F172A] mb-4">Attribution Analysis</h3>
                <p className="text-gray-500 text-sm">
                    Attribution by holding will be implemented when backend provides detailed time-series data.
                </p>
            </div>

            {/* Monthly Returns Placeholder */}
            <div className="bg-white border border-gray-200 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#0F172A] mb-4">Monthly Returns Heatmap</h3>
                <p className="text-gray-500 text-sm">
                    Monthly performance heatmap coming in next iteration.
                </p>
            </div>
        </div>
    );
}
