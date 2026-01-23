"use client";

import { useNexus } from "@/components/nexus/NexusProvider";
import { EmptyState } from "@/components/nexus/EmptyState";
import { SkeletonBlock } from "@/components/nexus/Skeleton";

export default function HoldingsPage() {
    const { state, status, error, mode, setMode } = useNexus();

    if (status === "error") {
        return (
            <EmptyState
                title="Unable to load holdings"
                description={error || "Check your backend connection and try again."}
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
                primaryAction={{ label: "Switch to Demo Mode", onClick: () => setMode("demo") }}
                secondaryAction={{ label: "Stay in Live Mode", onClick: () => setMode("live") }}
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

    const { holdings } = state;

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold text-[#0F172A] mb-2">Holdings</h2>
                <p className="text-gray-600">
                    {mode === "demo" ? "Demo portfolio positions" : "Current portfolio positions"}
                </p>
            </div>

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
                                        {holding.shares !== undefined ? holding.shares.toLocaleString() : "--"}
                                    </div>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-right">
                                    <div className="text-sm text-gray-900">
                                        {holding.price !== undefined ? `$${holding.price.toLocaleString()}` : "--"}
                                    </div>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-right">
                                    <div className="text-sm text-gray-900">
                                        {holding.weight !== undefined ? `${(holding.weight * 100).toFixed(1)}%` : "--"}
                                    </div>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-right">
                                    <div className="text-sm font-medium text-gray-900">
                                        {holding.value !== undefined ? `$${holding.value.toLocaleString()}` : "--"}
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
        </div>
    );
}
