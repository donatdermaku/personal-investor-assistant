"use client";

import { useNexus } from "@/components/nexus/NexusProvider";
import { definitionTooltip } from "@/lib/definitions";
import { MetricCard } from "@/components/ui/MetricCard";
import { getMetricReasons, getMetricStatus } from "@/lib/coverageLogic";
import { RunCreationModal } from "@/components/nexus/RunCreationModal";
import { useState } from "react";

export function MissionControl() {
    const { state, status, mode, setMode, openRunCreator, contextPanelOpen, toggleContextPanel } = useNexus();
    const [isCreatorOpen, setIsCreatorOpen] = useState(false);

    const handleOpenCreator = () => {
        setIsCreatorOpen(true);
    };

    const handleCloseCreator = () => {
        setIsCreatorOpen(false);
    };

    if (status === "loading" || !state) {
        // Loading state - Sidebar Skeleton
        return (
            <aside className="hidden lg:flex w-80 flex-col border-l border-[var(--color-nexus-border)] bg-[var(--color-nexus-surface)]/50 backdrop-blur-md h-screen sticky top-0 overflow-y-auto">
                <div className="p-6 border-b border-[var(--color-nexus-border)]">
                    <div className="h-6 w-32 bg-[var(--color-nexus-surface-hover)] animate-pulse rounded" />
                </div>
                <div className="p-6 space-y-6">
                    <div className="h-24 w-full bg-[var(--color-nexus-surface-hover)] animate-pulse rounded" />
                    <div className="h-24 w-full bg-[var(--color-nexus-surface-hover)] animate-pulse rounded" />
                </div>
            </aside>
        );
    }

    const { summary, manifest, coverage_summary, diagnostics } = state;
    // ... helpers ...
    const metricStatus = (kpiKey: string) => getMetricStatus(kpiKey, coverage_summary ?? null);

    const SidebarContent = () => (
        <>
            {/* HEADER */}
            <div className="p-6 border-b border-[var(--color-nexus-border)] flex items-center justify-between">
                <div>
                    <h2 className="text-sm font-sans font-bold text-[var(--color-nexus-text-secondary)] tracking-widest uppercase mb-1">
                        Mission Control
                    </h2>
                    <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${mode === "live" ? "bg-[var(--color-nexus-success)]" : "bg-[var(--color-nexus-primary)]"}`} />
                        <span className="font-mono text-xs text-[var(--color-nexus-text-main)]">
                            {mode === "live" ? "Live Portfolio" : "Demo Mode"}
                        </span>
                    </div>
                </div>
                {/* Mobile Close Button */}
                <button onClick={toggleContextPanel} className="lg:hidden text-[var(--color-nexus-text-muted)] hover:text-[var(--color-nexus-text-primary)]">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>

            {/* KPI SNAPSHOT */}
            <div className="p-6 space-y-6 border-b border-[var(--color-nexus-border)]">
                <div>
                    <h3 className="text-xs font-semibold text-[var(--color-nexus-text-secondary)] uppercase mb-3">
                        Key Performance
                    </h3>
                    <div className="space-y-3">
                        <div className="flex justify-between items-center">
                            <span className="text-sm text-[var(--color-nexus-text-muted)]">TWR</span>
                            <span className={`font-mono text-sm ${summary.twr && summary.twr >= 0 ? "text-[var(--color-nexus-success)]" : "text-[var(--color-nexus-danger)]"}`}>
                                {summary.twr !== null ? `${(summary.twr * 100).toFixed(2)}%` : "--"}
                            </span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-sm text-[var(--color-nexus-text-muted)]">Max Drawdown</span>
                            <span className="font-mono text-sm text-[var(--color-nexus-danger)]">
                                {summary.max_drawdown !== null ? `${(summary.max_drawdown * 100).toFixed(2)}%` : "--"}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* DIAGNOSTICS */}
            <div className="p-6 flex-1 overflow-y-auto">
                <h3 className="text-xs font-semibold text-[var(--color-nexus-text-secondary)] uppercase mb-3">
                    System Status
                </h3>
                {!diagnostics || diagnostics.length === 0 ? (
                    <div className="text-xs text-[var(--color-nexus-success)] flex items-center space-x-2">
                        <span>✓ All systems nominal.</span>
                    </div>
                ) : (
                    <div className="space-y-3">
                        {diagnostics.slice(0, 5).map((diag, idx) => (
                            <div key={idx} className="bg-[var(--color-nexus-surface)]/50 border border-[var(--color-nexus-border)] p-3 rounded text-xs">
                                <div className="flex items-start space-x-2">
                                    <span className={`flex-shrink-0 w-1.5 h-1.5 mt-1 rounded-full ${diag.severity === "critical" ? "bg-[var(--color-nexus-danger)]" : "bg-[var(--color-nexus-warning)]"}`} />
                                    <div>
                                        <span className="font-medium text-[var(--color-nexus-text-main)] block mb-0.5">{diag.check}</span>
                                        <span className="text-[var(--color-nexus-text-muted)]">{diag.message}</span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* ACTIONS */}
            <div className="p-6 border-t border-[var(--color-nexus-border)] bg-[var(--color-nexus-surface)]/50">
                <button
                    onClick={handleOpenCreator}
                    className="w-full py-2 px-4 bg-[var(--color-nexus-primary)]/10 hover:bg-[var(--color-nexus-primary)]/20 text-[var(--color-nexus-primary)] border border-[var(--color-nexus-primary)]/50 rounded text-sm font-medium transition-all"
                >
                    Create New Run
                </button>
                <div className="mt-3 flex justify-center">
                    <button
                        onClick={() => setMode(mode === "demo" ? "live" : "demo")}
                        className="text-xs text-[var(--color-nexus-text-muted)] hover:text-[var(--color-nexus-text-main)] underline decoration-dotted"
                    >
                        Switch to {mode === "demo" ? "Live" : "Demo"}
                    </button>
                </div>
            </div>
        </>
    );

    return (
        <>
            {/* DESKTOP SIDEBAR - Always visible on LG screens */}
            <aside className="hidden lg:flex w-80 flex-col border-l border-[var(--color-nexus-border)] bg-[var(--color-nexus-surface)]/30 backdrop-blur-md h-screen sticky top-0 overflow-y-auto z-40">
                <SidebarContent />
            </aside>

            {/* MOBILE SHEET - Visible on small screens when toggled */}
            {/* Backdrop */}
            {contextPanelOpen && (
                <div
                    className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 lg:hidden"
                    onClick={toggleContextPanel}
                />
            )}

            {/* Drawer */}
            <aside
                className={`
                    fixed inset-y-0 right-0 w-80 bg-[var(--color-nexus-bg)] border-l border-[var(--color-nexus-border)] z-50 lg:hidden transform transition-transform duration-300 ease-in-out flex flex-col
                    ${contextPanelOpen ? "translate-x-0" : "translate-x-full"}
                `}
            >
                <SidebarContent />
            </aside>

            {/* Modal for Run Creation - managed internally or via context if global */}
            {isCreatorOpen && (
                <RunCreationModal isOpen={isCreatorOpen} onClose={handleCloseCreator} />
            )}
        </>
    );

}
