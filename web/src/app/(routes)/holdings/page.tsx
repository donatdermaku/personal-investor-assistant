"use client";

import { useEffect, useState } from "react";
import { getNexusState } from "@/lib/api";
import { NexusState } from "@/types/nexus";

export default function HoldingsPage() {
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

    const { holdings } = state;

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold text-[#0F172A] mb-2">Holdings</h2>
                <p className="text-gray-600">Current portfolio positions</p>
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
