"use client";

import { useNexus } from "@/components/nexus/NexusProvider";
import { EmptyState } from "@/components/nexus/EmptyState";
import { SkeletonBlock } from "@/components/nexus/Skeleton";
import { SectionContext } from "@/components/nexus/SectionContext";
import { coverageLabel, coverageStatus } from "@/lib/coverage";

export default function HoldingsPage() {
    const { state, status, error, mode, setMode, openRunCreator } = useNexus();

    if (status === "error") {
        return (
            <EmptyState
                title="Unable to load holdings"
                description={error || "Check API connectivity, then retry in Live Mode or continue in Demo Mode."}
                primaryAction={{ label: "Retry in Live Mode", onClick: () => setMode("live") }}
                secondaryAction={{ label: "Switch to Demo Mode", onClick: () => setMode("demo") }}
            />
        );
    }

    if (status === "empty") {
        return (
            <EmptyState
                title="No holdings yet"
                description="Upload your portfolio or run a compute to see current positions."
                primaryAction={{ label: "Create Run", onClick: openRunCreator }}
                secondaryAction={{ label: "Switch to Demo Mode", onClick: () => setMode("demo") }}
            />
        );
    }

    if (status === "loading" || !state) {
        return (
            <div className="space-y-6">
                <SkeletonBlock className="h-8 w-40" />
                <SkeletonBlock className="h-4 w-56" />
                <div className="bg-white border border-gray-200 rounded-lg p-6">
                    <SkeletonBlock className="h-4 w-32" />
                    <SkeletonBlock className="mt-4 h-40 w-full" />
                </div>
            </div>
        );
    }

    const { holdings, summary, manifest, coverage_summary } = state;
    const coverage = coverageStatus(manifest, coverage_summary ?? null);
    const coverageText = coverageLabel(coverage);
    const asOf = summary.last_date || manifest.timestamp;
    const nonCashHoldings = holdings.filter((holding) => holding.ticker !== "CASH");

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold text-[#0F172A] mb-2">Holdings</h2>
                <p className="text-gray-600">
                    {mode === "demo" ? "Demo portfolio positions" : "Current portfolio positions"}
                </p>
                <div className="text-xs text-gray-400 mt-2">
                    {coverageText} · {asOf ? `As of ${asOf}` : "As of --"}
                </div>
            </div>

            <SectionContext
                title="Holdings Context"
                items={[
                    {
                        label: "What it measures",
                        text: "Current positions, weights, and market values from the latest available pricing.",
                    },
                    {
                        label: "Why it matters",
                        text: "Shows concentration and exposure that drive both returns and risk.",
                    },
                    {
                        label: "When it misleads",
                        text: "Missing prices or stale data can hide positions or misstate weights.",
                    },
                    {
                        label: "Assumptions",
                        text: "Holdings are priced with the latest end-of-day market data.",
                    },
                ]}
            />

            {/* Holdings Table */}
            <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                        <tr>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                                Ticker
                            </th>
                            <th className="px-6 py-3 text-right text-xs font-semibold text-gray-700 uppercase tracking-wider">
                                Shares
                            </th>
                            <th className="px-6 py-3 text-right text-xs font-semibold text-gray-700 uppercase tracking-wider">
                                Price
                            </th>
                            <th className="px-6 py-3 text-right text-xs font-semibold text-gray-700 uppercase tracking-wider">
                                Weight
                            </th>
                            <th className="px-6 py-3 text-right text-xs font-semibold text-gray-700 uppercase tracking-wider">
                                Value
                            </th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {holdings.map((holding, idx) => (
                            <tr key={idx} className="hover:bg-gray-50 transition-colors">
                                <td className="px-6 py-4 whitespace-nowrap">
                                    <div className="text-sm font-medium text-[#0F172A]">{holding.ticker}</div>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-right">
                                    <div className="text-sm text-gray-900">
                                        {holding.shares != null ? holding.shares.toLocaleString() : "--"}
                                    </div>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-right">
                                    <div className="text-sm text-gray-900">
                                        {holding.price != null ? `$${holding.price.toLocaleString()}` : "--"}
                                    </div>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-right">
                                    <div className="text-sm text-gray-900">
                                        {holding.weight != null ? `${(holding.weight * 100).toFixed(1)}%` : "--"}
                                    </div>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-right">
                                    <div className="text-sm font-medium text-gray-900">
                                        {holding.value != null ? `$${holding.value.toLocaleString()}` : "--"}
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {holdings.length === 0 && (
                <div className="text-center py-12 text-gray-500">
                    No holdings data available
                </div>
            )}

            {holdings.length > 0 && nonCashHoldings.length === 0 && (
                <div className="text-sm text-gray-500">
                    Holdings are cash-only for this run.
                </div>
            )}
        </div>
    );
}
