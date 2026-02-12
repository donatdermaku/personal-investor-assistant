"use client";

import { useMemo, useState } from "react";
import { downloadExport } from "@/lib/api";
import { useNexus } from "@/components/nexus/NexusProvider";
import type { NexusState } from "@/types/nexus";
import { SkeletonBlock } from "@/components/nexus/Skeleton";
import { coveragePercent } from "@/lib/coverage";

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

function coveragePercentFromState(state: NexusState | null) {
    return coveragePercent(state?.manifest || null, state?.coverage_summary || null);
}

export function ContextPanel() {
    const { state, status, error, mode, lastFetched, backendOk, benchmark, runs, runId, setRunId, contextPanelOpen, toggleContextPanel } = useNexus();
    const [toast, setToast] = useState<string | null>(null);

    const coverage = useMemo(() => coveragePercentFromState(state), [state]);
    const manifestRunId = state?.manifest.run_id;
    const exportsEnabled = Boolean(manifestRunId) && mode === "live";
    const loading = status === "loading";
    const hasError = status === "error";
    const portfolioCurrency = state?.portfolio?.currency ?? "--";

    const content = (
        <div className="space-y-8">
            {/* Portfolio Section */}
            <div>
                <div className="text-label mb-2">Active Portfolio</div>
                <div className="text-xl font-sans font-bold text-[var(--color-nexus-text-primary)] tracking-tight">
                    {state?.portfolio?.name || "--"}
                </div>
                <div className="flex items-center gap-2 mt-1">
                    <div className="text-xs font-mono text-[var(--color-nexus-text-secondary)]">
                        {state?.portfolio?.benchmark || benchmark || "--"}
                    </div>
                    <span className="text-[var(--color-nexus-border-light)]">•</span>
                    <div className="text-xs font-mono text-[var(--color-nexus-text-secondary)] uppercase">
                        {mode === "live" ? "Live Exec" : "Simulated"}
                    </div>
                </div>

                {runs.length > 0 && (
                    <div className="mt-4">
                        <div className="text-label mb-1">Select Run</div>
                        <select
                            className="w-full bg-[var(--color-nexus-surface-hover)] border border-[var(--color-nexus-border)] rounded-none text-xs font-mono text-[var(--color-nexus-text-primary)] px-2 py-1.5 focus:border-[var(--color-nexus-primary)] focus:outline-none transition-colors"
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

            {/* Run Stats Grid */}
            <div className="grid grid-cols-2 gap-px bg-[var(--color-nexus-border)] border border-[var(--color-nexus-border)]">
                <div className="bg-[var(--color-nexus-surface)] p-3 hover:bg-[var(--color-nexus-surface-hover)] transition-colors">
                    <div className="text-label mb-1">Last Run</div>
                    <div className="text-xs font-mono text-[var(--color-nexus-text-primary)] truncate">
                        {formatTimestamp(state?.manifest.timestamp).split(',')[0]}
                    </div>
                </div>
                <div className="bg-[var(--color-nexus-surface)] p-3 hover:bg-[var(--color-nexus-surface-hover)] transition-colors">
                    <div className="text-label mb-1">Coverage</div>
                    <div className={`text-xs font-mono font-bold ${typeof coverage === 'number' && coverage < 0.8 ? 'text-[var(--color-nexus-warning)]' : 'text-[var(--color-nexus-success)]'}`}>
                        {coverage === null ? "--" : `${(coverage * 100).toFixed(1)}%`}
                    </div>
                </div>
            </div>

            {/* Key KPIs */}
            <div>
                <div className="text-label mb-3">Key Metrics</div>
                <div className="space-y-1">
                    <div className="flex items-center justify-between p-3 bg-[var(--color-nexus-surface-hover)] border-l-2 border-[var(--color-nexus-primary)]">
                        <span className="text-xs text-[var(--color-nexus-text-secondary)]">TWR (YTD)</span>
                        <span className="font-mono font-bold text-[var(--color-nexus-text-primary)]">
                            {formatPercent(state?.summary.twr ?? null)}
                        </span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-[var(--color-nexus-surface-hover)] border-l-2 border-[var(--color-nexus-border-light)]">
                        <span className="text-xs text-[var(--color-nexus-text-secondary)]">Max Drawdown</span>
                        <span className="font-mono font-bold text-[var(--color-nexus-danger)]">
                            {formatPercent(state?.summary.max_drawdown ?? null)}
                        </span>
                    </div>
                </div>
            </div>

            {/* Exports */}
            <div>
                <div className="text-label mb-3 flex items-center justify-between">
                    <span>Data Export</span>
                    {!exportsEnabled && <span className="text-[10px] text-[var(--color-nexus-text-muted)]">Live only</span>}
                </div>

                <div className="flex flex-col gap-2">
                    {[
                        { id: "summary-json", label: "Summary JSON" },
                        { id: "performance-csv", label: "Performance CSV" },
                        { id: "monthly-returns-csv", label: "Monthly Returns" }
                    ].map((item) => (
                        <button
                            key={item.id}
                            type="button"
                            onClick={async () => {
                                if (!exportsEnabled || !manifestRunId) return;
                                try {
                                    await downloadExport(manifestRunId, item.id);
                                } catch (err) {
                                    setToast(err instanceof Error ? err.message : "Export failed.");
                                }
                            }}
                            disabled={!exportsEnabled}
                            className="group flex items-center justify-between w-full px-3 py-2 text-xs text-left border border-[var(--color-nexus-border)] hover:border-[var(--color-nexus-primary)] hover:bg-[var(--color-nexus-surface-hover)] transition-all disabled:opacity-30 disabled:hover:border-[var(--color-nexus-border)] disabled:hover:bg-transparent"
                        >
                            <span className="text-[var(--color-nexus-text-secondary)] group-hover:text-[var(--color-nexus-text-primary)] font-mono">{item.label}</span>
                            <span className="text-[var(--color-nexus-primary)] opacity-0 group-hover:opacity-100 transition-opacity">↓</span>
                        </button>
                    ))}
                </div>

                <div className="mt-4 pt-4 border-t border-[var(--color-nexus-border)] text-[10px] text-[var(--color-nexus-text-muted)] space-y-1 font-mono">
                    <div>Run ID: <span className="text-[var(--color-nexus-text-secondary)]">{manifestRunId ? manifestRunId.slice(0, 8) : "--"}</span></div>
                    <div>Currency: {portfolioCurrency}</div>
                    <div>Sources: Yahoo Finance, FRED</div>
                </div>
            </div>
        </div>
    );

    return (
        <>
            {/* Backdrop */}
            <div
                className={`fixed inset-0 bg-black/50 backdrop-blur-sm z-40 transition-opacity duration-300 ${contextPanelOpen ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"}`}
                onClick={toggleContextPanel}
                role="button"
                tabIndex={0}
                aria-label="Close panel"
            />

            {/* Floating HUD Panel */}
            <aside
                className={`
                    fixed right-0 top-0 h-screen w-80 
                    bg-[var(--color-nexus-surface)]/95 backdrop-blur-xl 
                    border-l border-[var(--color-nexus-border)] 
                    p-8 z-50 overflow-y-auto shadow-2xl
                    transition-transform duration-300 ease-out
                    ${contextPanelOpen ? "translate-x-0" : "translate-x-full"}
                `}
            >
                <div className="mb-8 flex items-center justify-between">
                    <span className="text-[10px] uppercase tracking-widest text-[var(--color-nexus-text-muted)]">Mission Control</span>
                    <button
                        onClick={toggleContextPanel}
                        className="text-[var(--color-nexus-text-secondary)] hover:text-[var(--color-nexus-primary)] transition-colors"
                    >
                        ✕
                    </button>
                </div>

                {loading ? (
                    <div className="space-y-4 opacity-50">
                        <SkeletonBlock className="h-8 w-1/2 bg-[var(--color-nexus-surface-hover)]" />
                        <SkeletonBlock className="h-24 w-full bg-[var(--color-nexus-surface-hover)]" />
                        <SkeletonBlock className="h-32 w-full bg-[var(--color-nexus-surface-hover)]" />
                    </div>
                ) : hasError ? (
                    <div className="p-4 border border-[var(--color-nexus-danger)] bg-[var(--color-nexus-danger)]/5 text-xs text-[var(--color-nexus-danger)]">
                        {error || "Unable to load context data."}
                    </div>
                ) : content}
            </aside>

            {toast && (
                <div className="fixed bottom-24 right-6 z-50 rounded-none border border-[var(--color-nexus-primary)] bg-[var(--color-nexus-surface)] px-4 py-3 text-xs text-[var(--color-nexus-text-primary)] shadow-[var(--shadow-glow)]">
                    {toast}
                    <button
                        type="button"
                        className="ml-3 text-[var(--color-nexus-primary)] hover:underline"
                        onClick={() => setToast(null)}
                    >
                        DISMISS
                    </button>
                </div>
            )}
        </>
    );
}
