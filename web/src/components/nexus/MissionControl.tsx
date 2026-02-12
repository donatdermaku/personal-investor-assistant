"use client";

import { useEffect, useState } from "react";
import { downloadExport } from "@/lib/api";
import { RunCreationModal } from "@/components/nexus/RunCreationModal";
import { useNexus } from "@/components/nexus/NexusProvider";
import { useMediaQuery } from "@/hooks/useMediaQuery";

function formatPercent(value: number | null | undefined) {
    if (value === null || value === undefined || Number.isNaN(value)) return "--";
    return `${(value * 100).toFixed(2)}%`;
}

function formatTimestamp(value?: string | null) {
    if (!value) return "--";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString();
}

export function MissionControl() {
    const {
        state,
        status,
        mode,
        setMode,
        runId,
        runs,
        setRunId,
        contextPanelOpen,
        toggleContextPanel,
        closeContextPanel,
    } = useNexus();
    const [isCreatorOpen, setIsCreatorOpen] = useState(false);
    const [toast, setToast] = useState<string | null>(null);
    const isDesktopMission = useMediaQuery("(min-width: 1200px)");
    const isPhone = useMediaQuery("(max-width: 767px)");
    const isTabletMission = !isDesktopMission && !isPhone;

    const loading = status === "loading";
    const manifestRunId = state?.manifest.run_id;
    const exportsEnabled = Boolean(manifestRunId) && mode === "live";
    const panelMode = contextPanelOpen ? "open" : "closed";

    useEffect(() => {
        if (isDesktopMission && contextPanelOpen) {
            closeContextPanel();
        }
    }, [isDesktopMission, contextPanelOpen, closeContextPanel]);

    const sectionContent = (
        <>
            <div className="p-6 border-b border-[var(--color-nexus-border)]">
                <h2 className="text-sm font-sans font-bold text-[var(--color-nexus-text-secondary)] tracking-widest uppercase mb-2">
                    Mission Control
                </h2>
                <div className="flex items-center justify-between gap-3">
                    <div className="text-xs font-mono text-[var(--color-nexus-text-secondary)]">
                        {mode === "live" ? "Live Portfolio" : "Demo Portfolio"}
                    </div>
                    <button
                        type="button"
                        onClick={() => setMode(mode === "demo" ? "live" : "demo")}
                        className="nexus-touch-target px-3 text-[11px] uppercase tracking-wider border border-[var(--color-nexus-border)] text-[var(--color-nexus-text-secondary)] hover:text-[var(--color-nexus-primary)] hover:border-[var(--color-nexus-primary)] transition-colors"
                    >
                        {mode === "demo" ? "Switch Live" : "Switch Demo"}
                    </button>
                </div>
            </div>

            <div className="p-6 border-b border-[var(--color-nexus-border)] space-y-4">
                <div>
                    <div className="text-label mb-2">Portfolio</div>
                    <div className="text-sm font-semibold text-[var(--color-nexus-text-primary)]">
                        {state?.portfolio?.name ?? "--"}
                    </div>
                </div>
                <div>
                    <div className="text-label mb-2">Last Run</div>
                    <div className="text-xs font-mono text-[var(--color-nexus-text-secondary)]">
                        {formatTimestamp(state?.manifest.timestamp)}
                    </div>
                </div>
                {runs.length > 0 && (
                    <div>
                        <div className="text-label mb-2">Run Selector</div>
                        <select
                            className="w-full bg-[var(--color-nexus-surface-hover)] border border-[var(--color-nexus-border)] text-xs font-mono text-[var(--color-nexus-text-primary)] px-3 py-2 focus:border-[var(--color-nexus-primary)] focus:outline-none"
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

            <div className="p-6 border-b border-[var(--color-nexus-border)] space-y-3">
                <div className="text-label">Key Metrics</div>
                <div className="nexus-card p-4">
                    <div className="flex items-center justify-between text-sm min-w-0 gap-3">
                        <span className="text-[var(--color-nexus-text-secondary)]">TWR</span>
                        <span className="font-mono text-[var(--color-nexus-text-primary)] nexus-inline-number text-right">{formatPercent(state?.summary.twr)}</span>
                    </div>
                </div>
                <div className="nexus-card p-4">
                    <div className="flex items-center justify-between text-sm min-w-0 gap-3">
                        <span className="text-[var(--color-nexus-text-secondary)]">Max Drawdown</span>
                        <span className="font-mono text-[var(--color-nexus-danger)] nexus-inline-number text-right">{formatPercent(state?.summary.max_drawdown)}</span>
                    </div>
                </div>
                <div className="nexus-card p-4">
                    <div className="flex items-center justify-between text-sm min-w-0 gap-3">
                        <span className="text-[var(--color-nexus-text-secondary)]">Portfolio Value</span>
                        <span className="font-mono text-[var(--color-nexus-text-primary)] nexus-inline-number text-right">
                            {state?.summary.final_value != null ? `$${state.summary.final_value.toLocaleString()}` : "--"}
                        </span>
                    </div>
                </div>
            </div>

            <div className="p-6 border-b border-[var(--color-nexus-border)] space-y-2">
                <div className="text-label mb-1">Export</div>
                {[
                    { id: "summary-json", label: "Summary JSON" },
                    { id: "performance-csv", label: "Performance CSV" },
                    { id: "monthly-returns-csv", label: "Monthly Returns CSV" },
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
                        className="nexus-touch-target w-full text-left border border-[var(--color-nexus-border)] px-3 py-2 text-xs text-[var(--color-nexus-text-secondary)] hover:text-[var(--color-nexus-text-primary)] hover:border-[var(--color-nexus-primary)] disabled:opacity-40 disabled:hover:border-[var(--color-nexus-border)] transition-colors"
                    >
                        {item.label}
                    </button>
                ))}
            </div>

            <div className="p-6 mt-auto space-y-3">
                <button
                    type="button"
                    onClick={() => setIsCreatorOpen(true)}
                    className="nexus-touch-target w-full py-2 px-4 bg-[var(--color-nexus-primary)]/10 hover:bg-[var(--color-nexus-primary)]/20 text-[var(--color-nexus-primary)] border border-[var(--color-nexus-primary)]/50 text-sm font-medium transition-colors"
                >
                    Create New Run
                </button>
                <button
                    type="button"
                    onClick={closeContextPanel}
                    className="nexus-touch-target w-full py-2 px-4 border border-[var(--color-nexus-border)] text-xs text-[var(--color-nexus-text-secondary)] hover:text-[var(--color-nexus-primary)] hover:border-[var(--color-nexus-primary)] transition-colors min-[1200px]:hidden"
                >
                    Close
                </button>
            </div>
        </>
    );

    return (
        <>
            {isDesktopMission && (
                <aside className="flex flex-col min-h-screen border-l border-[var(--color-nexus-border)] bg-[var(--color-nexus-surface)]/60 backdrop-blur-md nexus-panel-scroll">
                    {loading ? (
                        <div className="p-6 space-y-4">
                            <div className="h-5 w-36 bg-[var(--color-nexus-surface-hover)] animate-pulse" />
                            <div className="h-24 bg-[var(--color-nexus-surface-hover)] animate-pulse" />
                            <div className="h-24 bg-[var(--color-nexus-surface-hover)] animate-pulse" />
                        </div>
                    ) : (
                        sectionContent
                    )}
                </aside>
            )}

            {isTabletMission && (
                <>
                    <div
                        className={`fixed inset-0 bg-black/50 backdrop-blur-sm z-50 transition-opacity duration-300 ${panelMode === "open" ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"}`}
                        onClick={closeContextPanel}
                    />
                    <aside
                        role="dialog"
                        aria-modal="true"
                        aria-label="Mission Control panel"
                        className={`fixed inset-y-0 right-0 w-[340px] max-w-[85vw] bg-[var(--color-nexus-bg)] border-l border-[var(--color-nexus-border)] z-[60] transform transition-transform duration-300 flex flex-col nexus-panel-scroll ${panelMode === "open" ? "translate-x-0" : "translate-x-full"}`}
                    >
                        {sectionContent}
                    </aside>
                </>
            )}

            {isPhone && (
                <>
                    <div
                        className={`fixed inset-0 bg-black/45 backdrop-blur-sm z-50 transition-opacity duration-300 ${panelMode === "open" ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"}`}
                        onClick={closeContextPanel}
                    />
                    <aside
                        role="dialog"
                        aria-modal="true"
                        aria-label="Mission Control panel"
                        className={`fixed bottom-0 left-0 right-0 z-[60] rounded-t-xl border-t border-[var(--color-nexus-border)] bg-[var(--color-nexus-bg)] max-h-[85vh] min-h-[48vh] nexus-panel-scroll transform transition-transform duration-300 ${panelMode === "open" ? "translate-y-0" : "translate-y-full"}`}
                    >
                        {sectionContent}
                    </aside>
                </>
            )}

            {isPhone && (
                <button
                    type="button"
                    aria-label="Open Mission Control"
                    onClick={toggleContextPanel}
                    className="fixed bottom-24 right-4 z-40 nexus-touch-target px-4 py-3 border border-[var(--color-nexus-primary)] bg-[var(--color-nexus-primary)] text-black text-xs font-mono uppercase tracking-wide shadow-[var(--shadow-glow)]"
                >
                    Mission
                </button>
            )}

            {toast && (
                <div className="fixed bottom-24 right-6 z-[70] border border-[var(--color-nexus-primary)] bg-[var(--color-nexus-surface)] px-4 py-3 text-xs text-[var(--color-nexus-text-primary)]">
                    {toast}
                    <button
                        type="button"
                        aria-label="Dismiss notification"
                        className="ml-3 text-[var(--color-nexus-primary)] hover:underline"
                        onClick={() => setToast(null)}
                    >
                        Dismiss
                    </button>
                </div>
            )}

            {isCreatorOpen && <RunCreationModal isOpen={isCreatorOpen} onClose={() => setIsCreatorOpen(false)} />}
        </>
    );
}
