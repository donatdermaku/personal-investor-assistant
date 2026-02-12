"use client";

import { useNexus } from "@/components/nexus/NexusProvider";
import { useMediaQuery } from "@/hooks/useMediaQuery";

function formatTimestamp(value: string | null) {
    if (!value) return "--";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export function TopBar() {
    const {
        mode,
        setMode,
        portfolioId,
        setPortfolioId,
        benchmark,
        setBenchmark,
        status,
        loadingProgress,
        lastFetched,
        backendOk,
        refresh,
        openRunCreator,
        contextPanelOpen,
        toggleContextPanel,
        toggleNav,
    } = useNexus();
    const isDesktopMission = useMediaQuery("(min-width: 1200px)");
    const showMissionButton = !isDesktopMission;

    return (
        <div className="mb-8 md:mb-10 flex flex-col xl:flex-row xl:items-end xl:justify-between gap-6 border-b border-[var(--color-nexus-border)] pb-6 relative">
            <div className="flex flex-col gap-1">
                <div className="text-label mb-1">System Status</div>
                <div className="flex items-center gap-4">
                    <button
                        type="button"
                        onClick={toggleNav}
                        className="nexus-touch-target inline-flex items-center justify-center border border-[var(--color-nexus-border)] px-2.5 text-[var(--color-nexus-text-secondary)] hover:text-[var(--color-nexus-primary)] hover:border-[var(--color-nexus-primary)] transition-colors lg:hidden"
                        aria-label="Open navigation"
                    >
                        ☰
                    </button>
                    <div className="flex items-center gap-2">
                        <div className={`w-1.5 h-1.5 rounded-full ${backendOk ? "bg-[var(--color-nexus-success)] shadow-[0_0_10px_var(--color-nexus-success)]" : "bg-[var(--color-nexus-danger)]"}`} />
                        <span className="text-sm font-mono text-[var(--color-nexus-text-primary)]">
                            {backendOk ? "ONLINE" : "OFFLINE"}
                        </span>
                    </div>
                    <span className="text-[var(--color-nexus-border-light)]">|</span>
                    <div className="text-sm font-mono text-[var(--color-nexus-text-secondary)]">
                        LAST SYNC: <span className="text-[var(--color-nexus-text-primary)]">{formatTimestamp(lastFetched)}</span>
                    </div>
                </div>
            </div>

            <div className="flex flex-wrap items-center gap-4 md:gap-6">

                {/* Mode Switcher */}
                <div className="flex items-center bg-[var(--color-nexus-surface)] rounded-sm p-1 border border-[var(--color-nexus-border)]">
                    <button
                        type="button"
                        onClick={() => setMode("live")}
                        className={`px-4 py-1.5 text-xs font-mono uppercase tracking-wider transition-all ${mode === "live"
                            ? "bg-[var(--color-nexus-border-light)] text-[var(--color-nexus-text-primary)]"
                            : "text-[var(--color-nexus-text-muted)] hover:text-[var(--color-nexus-text-secondary)]"
                            }`}
                    >
                        Live
                    </button>
                    <button
                        type="button"
                        onClick={() => setMode("demo")}
                        className={`px-4 py-1.5 text-xs font-mono uppercase tracking-wider transition-all ${mode === "demo"
                            ? "bg-[var(--color-nexus-border-light)] text-[var(--color-nexus-text-primary)]"
                            : "text-[var(--color-nexus-text-muted)] hover:text-[var(--color-nexus-text-secondary)]"
                            }`}
                    >
                        Demo
                    </button>
                </div>

                <div className="h-8 w-[1px] bg-[var(--color-nexus-border)] hidden xl:block" />

                {/* Controls */}
                <div className="flex items-center gap-4">
                    <div className="group flex flex-col">
                        <label className="text-[10px] uppercase tracking-wider text-[var(--color-nexus-text-muted)] mb-1 group-focus-within:text-[var(--color-nexus-primary)] transition-colors">Portfolio</label>
                        <input
                            className="bg-transparent border-b border-[var(--color-nexus-border)] py-1 text-sm font-mono text-[var(--color-nexus-text-primary)] focus:outline-none focus:border-[var(--color-nexus-primary)] transition-colors w-24"
                            value={portfolioId}
                            onChange={(event) => setPortfolioId(event.target.value)}
                        />
                    </div>

                    <div className="group flex flex-col">
                        <label className="text-[10px] uppercase tracking-wider text-[var(--color-nexus-text-muted)] mb-1 group-focus-within:text-[var(--color-nexus-primary)] transition-colors">Ref</label>
                        <input
                            className="bg-transparent border-b border-[var(--color-nexus-border)] py-1 text-sm font-mono text-[var(--color-nexus-text-primary)] focus:outline-none focus:border-[var(--color-nexus-primary)] transition-colors w-16"
                            value={benchmark}
                            onChange={(event) => setBenchmark(event.target.value.toUpperCase())}
                        />
                    </div>
                </div>

                <button
                    type="button"
                    onClick={refresh}
                    disabled={status === "loading"}
                    className="group relative ml-4"
                >
                    <div className="absolute -inset-2 bg-gradient-to-r from-[var(--color-nexus-primary)] to-[var(--color-nexus-accent)] rounded-lg opacity-20 group-hover:opacity-40 blur transition duration-200" />
                    <div className="relative border border-[var(--color-nexus-border)] bg-[var(--color-nexus-bg)] px-4 py-2 text-xs font-mono uppercase tracking-widest text-[var(--color-nexus-text-primary)] hover:text-[var(--color-nexus-primary)] transition-colors">
                        {status === "loading" ? "SYNCING..." : "REFRESH"}
                    </div>
                </button>

                <button
                    type="button"
                    onClick={openRunCreator}
                    className="border border-[var(--color-nexus-primary)] bg-[var(--color-nexus-primary)]/10 px-4 py-2 text-xs font-mono uppercase tracking-widest text-[var(--color-nexus-primary)] hover:bg-[var(--color-nexus-primary)] hover:text-black transition-all duration-300"
                >
                    NEW RUN
                </button>

                {showMissionButton && (
                    <button
                        type="button"
                        onClick={toggleContextPanel}
                        className={`
                            border px-4 py-2 text-xs font-mono uppercase tracking-widest transition-all duration-300 hidden md:inline-flex
                            ${contextPanelOpen
                                ? "border-[var(--color-nexus-primary)] bg-[var(--color-nexus-primary)] text-black"
                                : "border-[var(--color-nexus-border)] text-[var(--color-nexus-text-secondary)] hover:text-[var(--color-nexus-primary)] hover:border-[var(--color-nexus-primary)]"
                            }
                        `}
                    >
                        MISSION CONTROL
                    </button>
                )}
            </div>

            {status === "loading" && (
                <div className="absolute bottom-0 left-0 w-full">
                    <div className="h-[2px] w-full bg-[var(--color-nexus-border)] overflow-hidden">
                        <div
                            className="h-full bg-[var(--color-nexus-primary)] transition-all duration-300 shadow-[0_0_10px_var(--color-nexus-primary)]"
                            style={{ width: `${Math.max(5, Math.min(100, Math.round(loadingProgress || 0)))}%` }}
                        />
                    </div>
                </div>
            )}
        </div>
    );
}
