"use client";

import { useEffect, useMemo, useState } from "react";
import { getNexusState } from "@/lib/api";
import type { NexusState } from "@/types/nexus";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
    const [state, setState] = useState<NexusState | null>(null);
    const [loading, setLoading] = useState(true);
    const [open, setOpen] = useState(false);

    useEffect(() => {
        getNexusState().then((data) => {
            setState(data);
            setLoading(false);
        });
    }, []);

    const coverage = useMemo(() => coveragePercent(state), [state]);
    const runId = state?.manifest.run_id;

    const content = (
        <div className="space-y-4">
            <div>
                <div className="text-xs uppercase tracking-wider text-gray-400">Portfolio</div>
                <div className="text-base font-semibold text-[#0F172A]">
                    {state?.portfolio?.name || "--"}
                </div>
                <div className="text-xs text-gray-500">
                    Benchmark: {state?.portfolio?.benchmark || "--"}
                </div>
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
                <div className="text-xs uppercase tracking-wider text-gray-400 mb-2">Key KPIs</div>
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
                <div className="text-xs uppercase tracking-wider text-gray-400">Export</div>
                <div className="flex flex-col gap-2 text-sm">
                    <a
                        className="rounded-md border border-gray-200 px-3 py-2 text-gray-700 hover:bg-gray-50"
                        href={runId ? `${API_BASE_URL}/run/${runId}/export/summary` : "#"}
                    >
                        Summary JSON
                    </a>
                    <a
                        className="rounded-md border border-gray-200 px-3 py-2 text-gray-700 hover:bg-gray-50"
                        href={runId ? `${API_BASE_URL}/run/${runId}/export/performance` : "#"}
                    >
                        Performance CSV
                    </a>
                    <a
                        className="rounded-md border border-gray-200 px-3 py-2 text-gray-700 hover:bg-gray-50"
                        href={runId ? `${API_BASE_URL}/run/${runId}/export/monthly-returns` : "#"}
                    >
                        Monthly Returns
                    </a>
                </div>
            </div>
        </div>
    );

    return (
        <>
            <aside className="w-80 bg-white border-l border-[#E5E7EB] h-screen fixed right-0 top-0 p-6 hidden lg:block">
                <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wider mb-4">
                    Context
                </h3>
                {loading ? <div className="text-xs text-gray-400">Loading...</div> : content}
            </aside>

            <div className="lg:hidden fixed right-4 bottom-16 z-20">
                <button
                    type="button"
                    onClick={() => setOpen((prev) => !prev)}
                    className="rounded-full bg-[#0F172A] text-white px-4 py-2 text-xs uppercase tracking-wider shadow-lg"
                >
                    {open ? "Close" : "Context"}
                </button>
            </div>

            {open && (
                <div className="lg:hidden fixed inset-x-0 bottom-0 z-10 bg-white border-t border-[#E5E7EB] p-6 shadow-2xl max-h-[70vh] overflow-y-auto">
                    <div className="text-sm font-semibold text-gray-900 uppercase tracking-wider mb-4">
                        Context
                    </div>
                    {loading ? <div className="text-xs text-gray-400">Loading...</div> : content}
                </div>
            )}
        </>
    );
}
