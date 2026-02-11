"use client";

import { useEffect, useState } from "react";
import { EmptyState } from "@/components/nexus/EmptyState";
import { getOpsHealth } from "@/lib/api";
import type { OpsHealthResponse } from "@/types/nexus";

function formatTimestamp(value: string | null | undefined) {
    if (!value) return "--";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString();
}

function formatUptime(seconds: number) {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hrs}h ${mins}m ${secs}s`;
}

export default function OperationsPage() {
    const [data, setData] = useState<OpsHealthResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const load = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await getOpsHealth();
            setData(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load operations status.");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void load();
    }, []);

    if (error) {
        return (
            <EmptyState
                title="Unable to load operations dashboard"
                description={error}
                primaryAction={{ label: "Retry", onClick: () => void load() }}
            />
        );
    }

    return (
        <div className="space-y-8">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-3xl font-bold text-[#0F172A] mb-2">Operations</h2>
                    <p className="text-gray-600">Backend health, cache, and rate limit posture.</p>
                </div>
                <button
                    type="button"
                    onClick={() => void load()}
                    disabled={loading}
                    className="rounded-md border border-gray-200 bg-white px-3 py-2 text-sm font-semibold text-gray-700 hover:bg-gray-50 disabled:opacity-60"
                >
                    {loading ? "Refreshing..." : "Refresh"}
                </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                <div className="nexus-card">
                    <div className="text-xs uppercase tracking-wider text-gray-400">API Status</div>
                    <div className="mt-2 text-xl font-semibold text-[#0F172A]">{data?.status || "--"}</div>
                </div>
                <div className="nexus-card">
                    <div className="text-xs uppercase tracking-wider text-gray-400">Uptime</div>
                    <div className="mt-2 text-xl font-semibold text-[#0F172A]">
                        {data ? formatUptime(data.runtime.uptime_seconds) : "--"}
                    </div>
                </div>
                <div className="nexus-card">
                    <div className="text-xs uppercase tracking-wider text-gray-400">Memory (RSS)</div>
                    <div className="mt-2 text-xl font-semibold text-[#0F172A]">
                        {data ? `${data.runtime.rss_mb.toFixed(1)} MB` : "--"}
                    </div>
                </div>
                <div className="nexus-card">
                    <div className="text-xs uppercase tracking-wider text-gray-400">Database</div>
                    <div className="mt-2 text-xl font-semibold text-[#0F172A]">
                        {data ? `${data.database.status} (${data.database.backend})` : "--"}
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="nexus-card space-y-3">
                    <h3 className="text-lg font-semibold text-[#0F172A]">Rate Limit</h3>
                    <div className="text-sm text-gray-600">
                        Enabled: <span className="font-semibold text-gray-800">{data?.rate_limit.enabled ? "Yes" : "No"}</span>
                    </div>
                    <div className="text-sm text-gray-600">
                        Window: <span className="font-semibold text-gray-800">{data?.rate_limit.window_seconds ?? "--"}s</span>
                    </div>
                    <div className="text-sm text-gray-600">
                        Limit: <span className="font-semibold text-gray-800">{data?.rate_limit.limit_per_window ?? "--"} requests/window</span>
                    </div>
                </div>

                <div className="nexus-card space-y-3">
                    <h3 className="text-lg font-semibold text-[#0F172A]">Latest Run</h3>
                    {data?.latest_run ? (
                        <>
                            <div className="text-sm text-gray-600">
                                Run ID: <span className="font-semibold text-gray-800">{data.latest_run.run_id}</span>
                            </div>
                            <div className="text-sm text-gray-600">
                                Status: <span className="font-semibold text-gray-800">{data.latest_run.status}</span>
                            </div>
                            <div className="text-sm text-gray-600">
                                Timestamp: <span className="font-semibold text-gray-800">{formatTimestamp(data.latest_run.timestamp)}</span>
                            </div>
                        </>
                    ) : (
                        <div className="text-sm text-gray-500">No run metadata available.</div>
                    )}
                </div>
            </div>

            <div className="nexus-card">
                <h3 className="text-lg font-semibold text-[#0F172A] mb-2">Runtime Timestamps</h3>
                <div className="text-sm text-gray-600">Started: {formatTimestamp(data?.runtime.started_at)}</div>
                <div className="text-sm text-gray-600">Now: {formatTimestamp(data?.runtime.now)}</div>
            </div>
        </div>
    );
}
