"use client";

import { useMemo, useState } from "react";
import { downloadExport } from "@/lib/api";
import { useNexus } from "@/components/nexus/NexusProvider";
import type { NexusState } from "@/types/nexus";
import { SkeletonBlock } from "@/components/nexus/Skeleton";

function formatPercent(value: number | null) {
    if (value === null || Number.isNaN(value)) return "--";
    return `${(value * 100).toFixed(2)}%`;
}

function formatTimestamp(value?: string | null) {
    if (!value) return "--";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString();
}

function coveragePercent(state: NexusState | null) {
    const summary = state?.manifest.coverage_summary;
    if (!summary || Object.keys(summary).length === 0) return null;
    let covered = 0;
    let total = 0;
    Object.values(summary).forEach((item) => {
        covered += item.covered || 0;
        total += item.total || 0;
    });
    if (total <= 0) return null;
    return covered / total;
}

export function ContextPanel() {
    const { state, status, mode, lastFetched, backendOk, benchmark, runs, runId, setRunId } = useNexus();
    const [toast, setToast] = useState<string | null>(null);

    const coverage = useMemo(() => coveragePercent(state), [state]);
    const manifestRunId = state?.manifest.run_id;
    const exportsEnabled = Boolean(manifestRunId) && mode === "live";
    const loading = status === "loading";
    const hasError = status === "error";

    const content = (
        <div className="space-y-4">
            <div>
                <div className="inline-flex items-center rounded-full bg-[#F2F6FF] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-[#1E40AF]">
                    Portfolio
                </div>
                <div className="text-base font-semibold text-[#0F172A]">
                    {state?.portfolio?.name || "--"}
                </div>
                <div className="text-xs text-gray-500">
                    Benchmark: {state?.portfolio?.benchmark || benchmark || "--"}
                </div>
                <div className="text-xs text-gray-400 mt-1">
                    Mode: {mode === "live" ? "Live" : "Demo"}
                </div>
                {runs.length > 0 && (
                    <div className="mt-3">
                        <div className="text-xs uppercase tracking-wider text-gray-400">Run</div>
                        <select
                            className="mt-1 w-full rounded-md border border-gray-200 px-2 py-1 text-sm text-gray-700"
                            value={runId ?? runs[0]?.run_id}
                            onChange={(event) => setRunId(event.target.value)}
                        >
                            {runs.map((run) => (
                                <option key={run.run_id} value={run.run_id}>
                                    {run.run_id.slice(0, 8)} · {run.timestamp ? new Date(run.timestamp).toLocaleString() : "pending"}
                                </option>
                            ))}
                        </select>
                    </div>
                )}
            </div>

            <div className="grid grid-cols-2 gap-3">
                <div className="nexus-card bg-gray-50">
                    <div className="text-xs text-gray-500">Last Run</div>
                    <div className="text-sm font-semibold text-gray-900">
                        {formatTimestamp(state?.manifest.timestamp)}
                    </div>
                </div>
                <div className="nexus-card bg-gray-50">
                    <div className="text-xs text-gray-500">Data Coverage</div>
                    <div className="text-sm font-semibold text-gray-900">
                        {coverage === null ? "--" : `${(coverage * 100).toFixed(1)}%`}
                    </div>
                </div>
            </div>

            <div className="nexus-card">
                <div className="inline-flex items-center rounded-full bg-[#F2F6FF] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-[#1E40AF] mb-2">
                    Key KPIs
                </div>
                <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-500">TWR</span>
                    <span className="font-semibold text-gray-900">
                        {formatPercent(state?.summary.twr ?? null)}
                    </span>
                </div>
                <div className="flex items-center justify-between text-sm mt-2">
                    <span className="text-gray-500">Max Drawdown</span>
                    <span className="font-semibold text-gray-900">
                        {formatPercent(state?.summary.max_drawdown ?? null)}
                    </span>
                </div>
            </div>

            <div className="space-y-2">
                <div className="inline-flex items-center rounded-full bg-[#F2F6FF] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-[#1E40AF]">
                    Export
                </div>
                <div className="flex flex-col gap-2 text-sm">
                    <button
                        type="button"
                        onClick={async () => {
                            if (!exportsEnabled || !manifestRunId) return;
                            try {
                                await downloadExport(manifestRunId, "summary-json");
                            } catch (err) {
                                setToast(err instanceof Error ? err.message : "Export failed.");
                            }
                        }}
                        disabled={!exportsEnabled}
                        className="rounded-md border border-gray-200 px-3 py-2 text-left text-gray-700 hover:bg-gray-50 disabled:opacity-50"
                    >
                        Summary JSON
                    </button>
                    <button
                        type="button"
                        onClick={async () => {
                            if (!exportsEnabled || !manifestRunId) return;
                            try {
                                await downloadExport(manifestRunId, "performance-csv");
                            } catch (err) {
                                setToast(err instanceof Error ? err.message : "Export failed.");
                            }
                        }}
                        disabled={!exportsEnabled}
                        className="rounded-md border border-gray-200 px-3 py-2 text-left text-gray-700 hover:bg-gray-50 disabled:opacity-50"
                    >
                        Performance CSV
                    </button>
                    <button
                        type="button"
                        onClick={async () => {
                            if (!exportsEnabled || !manifestRunId) return;
                            try {
                                await downloadExport(manifestRunId, "monthly-returns-csv");
                            } catch (err) {
                                setToast(err instanceof Error ? err.message : "Export failed.");
                            }
                        }}
                        disabled={!exportsEnabled}
                        className="rounded-md border border-gray-200 px-3 py-2 text-left text-gray-700 hover:bg-gray-50 disabled:opacity-50"
                    >
                        Monthly Returns
                    </button>
                </div>
                {!exportsEnabled && (
                    <div className="text-xs text-gray-400">
                        Exports are available after a live run completes.
                    </div>
                )}
            </div>

            <div className="text-xs text-gray-400">
                Backend: {backendOk ? "Connected" : "Offline"} · Last fetch {formatTimestamp(lastFetched)}
            </div>
        </div>
    );

    return (
        <>
            <aside className="w-80 bg-white border-l border-[#E5E7EB] h-screen fixed right-0 top-0 p-6 hidden lg:block">
                <h3 className="inline-flex items-center rounded-full bg-[#F2F6FF] px-3 py-1 text-xs font-semibold uppercase tracking-wider text-[#1E40AF] mb-4">
                    Context
                </h3>
                {loading ? (
                    <div className="space-y-3">
                        <SkeletonBlock className="h-4 w-32" />
                        <SkeletonBlock className="h-10 w-full" />
                        <SkeletonBlock className="h-24 w-full" />
                        <SkeletonBlock className="h-32 w-full" />
                    </div>
                ) : hasError ? (
                    <div className="text-sm text-red-500">Unable to load context data.</div>
                ) : content}
            </aside>

            <div className="lg:hidden mt-8 px-6 pb-8">
                <div className="rounded-xl border border-[#E5E7EB] bg-white p-6 shadow-sm">
                    <div className="inline-flex items-center rounded-full bg-[#F2F6FF] px-3 py-1 text-xs font-semibold uppercase tracking-wider text-[#1E40AF] mb-4">
                        Context
                    </div>
                    {loading ? (
                        <div className="space-y-3">
                            <SkeletonBlock className="h-4 w-32" />
                            <SkeletonBlock className="h-10 w-full" />
                            <SkeletonBlock className="h-24 w-full" />
                            <SkeletonBlock className="h-32 w-full" />
                        </div>
                    ) : hasError ? (
                        <div className="text-sm text-red-500">Unable to load context data.</div>
                    ) : content}
                </div>
            </div>

            {toast && (
                <div className="fixed bottom-24 right-6 z-30 rounded-md bg-[#0F172A] px-4 py-2 text-xs text-white shadow-lg">
                    {toast}
                    <button
                        type="button"
                        className="ml-3 text-gray-200"
                        onClick={() => setToast(null)}
                    >
                        Close
                    </button>
                </div>
            )}
        </>
    );
}
