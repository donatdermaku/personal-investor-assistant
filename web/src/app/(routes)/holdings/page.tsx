"use client";

import { useEffect, useRef, useState } from "react";
import { useNexus } from "@/components/nexus/NexusProvider";
import { EmptyState } from "@/components/nexus/EmptyState";
import { SkeletonBlock } from "@/components/nexus/Skeleton";
import { SectionContext } from "@/components/nexus/SectionContext";
import { coverageLabel, coverageStatus } from "@/lib/coverage";

export default function HoldingsPage() {
    const { state, status, error, mode, setMode, openRunCreator } = useNexus();
    const [expandedTicker, setExpandedTicker] = useState<string | null>(null);
    const desktopScrollRef = useRef<HTMLDivElement | null>(null);
    const [scrollTop, setScrollTop] = useState(0);
    const [viewportHeight, setViewportHeight] = useState(640);
    const holdings = state?.holdings ?? [];
    const rowHeight = 72;
    const overscan = 10;
    const virtualizationEnabled = holdings.length > 140;
    const startIndex = virtualizationEnabled ? Math.max(0, Math.floor(scrollTop / rowHeight) - overscan) : 0;
    const visibleCount = virtualizationEnabled
        ? Math.ceil(viewportHeight / rowHeight) + overscan * 2
        : holdings.length;
    const endIndex = virtualizationEnabled ? Math.min(holdings.length, startIndex + visibleCount) : holdings.length;
    const visibleHoldings = holdings.slice(startIndex, endIndex);
    const topPad = virtualizationEnabled ? startIndex * rowHeight : 0;
    const bottomPad = virtualizationEnabled ? Math.max(0, (holdings.length - endIndex) * rowHeight) : 0;

    useEffect(() => {
        const node = desktopScrollRef.current;
        if (!node) return;
        const updateHeight = () => setViewportHeight(node.clientHeight || 640);
        updateHeight();
        const resizeObserver = new ResizeObserver(updateHeight);
        resizeObserver.observe(node);
        return () => resizeObserver.disconnect();
    }, []);

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
            <div className="nexus-page">
                <SkeletonBlock className="h-8 w-40 bg-[var(--color-nexus-surface)]" />
                <SkeletonBlock className="h-4 w-56 bg-[var(--color-nexus-surface)]" />
                <div className="nexus-card p-6 min-h-[400px]">
                    <SkeletonBlock className="h-4 w-32 bg-[var(--color-nexus-surface-hover)]" />
                    <SkeletonBlock className="mt-4 h-full w-full bg-[var(--color-nexus-surface-hover)]" />
                </div>
            </div>
        );
    }

    const { summary, manifest, coverage_summary } = state;
    const coverage = coverageStatus(manifest, coverage_summary ?? null);
    const coverageText = coverageLabel(coverage);
    const asOf = summary.last_date || manifest.timestamp;
    const nonCashHoldings = holdings.filter((holding) => holding.ticker !== "CASH");

    return (
        <div className="nexus-page animate-in fade-in duration-500">
            <div>
                <h2 className="nexus-heading-1 font-sans font-bold text-[var(--color-nexus-text-primary)] mb-2 tracking-tight">Holdings</h2>
                <p className="text-[var(--color-nexus-text-secondary)] text-base">
                    {mode === "demo" ? "Demo portfolio positions" : "Current portfolio positions"}
                </p>
                <div className="text-xs text-[var(--color-nexus-text-muted)] mt-2 font-mono">
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

            <section className="nexus-card overflow-hidden hidden md:block">
                <div
                    ref={desktopScrollRef}
                    className="max-h-[68vh] overflow-auto"
                    onScroll={(event) => {
                        if (!virtualizationEnabled) return;
                        setScrollTop(event.currentTarget.scrollTop);
                    }}
                >
                    <table className="min-w-full divide-y divide-[var(--color-nexus-border)] text-sm">
                        <thead className="bg-[var(--color-nexus-surface-hover)] sticky top-0 z-10">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-[var(--color-nexus-text-secondary)] uppercase tracking-wider font-mono">Ticker</th>
                                <th className="px-6 py-3 text-right text-xs font-semibold text-[var(--color-nexus-text-secondary)] uppercase tracking-wider font-mono">Shares</th>
                                <th className="px-6 py-3 text-right text-xs font-semibold text-[var(--color-nexus-text-secondary)] uppercase tracking-wider font-mono min-[1200px]:table-cell hidden">Price</th>
                                <th className="px-6 py-3 text-right text-xs font-semibold text-[var(--color-nexus-text-secondary)] uppercase tracking-wider font-mono">Weight</th>
                                <th className="px-6 py-3 text-right text-xs font-semibold text-[var(--color-nexus-text-secondary)] uppercase tracking-wider font-mono">Value</th>
                            </tr>
                        </thead>
                        <tbody className="bg-[var(--color-nexus-surface)] divide-y divide-[var(--color-nexus-border)]">
                            {topPad > 0 && (
                                <tr aria-hidden="true">
                                    <td colSpan={5} style={{ height: `${topPad}px`, padding: 0, border: 0 }} />
                                </tr>
                            )}
                            {visibleHoldings.map((holding, localIdx) => {
                                const idx = startIndex + localIdx;
                                return (
                                <tr
                                    key={`${holding.ticker}-${idx}`}
                                    className="hover:bg-[var(--color-nexus-surface-hover)] transition-colors"
                                    style={virtualizationEnabled ? { height: `${rowHeight}px` } : undefined}
                                >
                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-[var(--color-nexus-text-primary)] font-mono">{holding.ticker}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-[var(--color-nexus-text-secondary)] font-mono">
                                        {holding.shares != null ? holding.shares.toLocaleString() : "--"}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-[var(--color-nexus-text-secondary)] font-mono min-[1200px]:table-cell hidden">
                                        {holding.price != null ? `$${holding.price.toLocaleString()}` : "--"}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-[var(--color-nexus-text-primary)] font-mono font-semibold">
                                        {holding.weight != null ? `${(holding.weight * 100).toFixed(1)}%` : "--"}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium text-[var(--color-nexus-text-primary)] font-mono">
                                        {holding.value != null ? `$${holding.value.toLocaleString()}` : "--"}
                                        <div className="min-[1200px]:hidden text-[10px] text-[var(--color-nexus-text-muted)] mt-0.5">
                                            Price: {holding.price != null ? `$${holding.price.toLocaleString()}` : "--"}
                                        </div>
                                    </td>
                                </tr>
                            )})}
                            {bottomPad > 0 && (
                                <tr aria-hidden="true">
                                    <td colSpan={5} style={{ height: `${bottomPad}px`, padding: 0, border: 0 }} />
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
                {virtualizationEnabled && (
                    <div className="px-6 py-3 border-t border-[var(--color-nexus-border)] text-xs text-[var(--color-nexus-text-muted)] font-mono">
                        Virtualized rows enabled for {holdings.length.toLocaleString()} holdings.
                    </div>
                )}
            </section>

            <section className="md:hidden grid gap-4">
                {holdings.map((holding, idx) => {
                    const expanded = expandedTicker === `${holding.ticker}-${idx}`;
                    return (
                        <article key={`${holding.ticker}-${idx}`} className="nexus-card p-4">
                            <button
                                type="button"
                                className="w-full text-left flex items-center justify-between nexus-touch-target"
                                onClick={() => setExpandedTicker(expanded ? null : `${holding.ticker}-${idx}`)}
                            >
                                <div>
                                    <div className="text-base font-semibold font-mono text-[var(--color-nexus-text-primary)]">{holding.ticker}</div>
                                    <div className="text-xs text-[var(--color-nexus-text-secondary)] mt-1">
                                        {holding.value != null ? `$${holding.value.toLocaleString()}` : "--"} · {holding.weight != null ? `${(holding.weight * 100).toFixed(1)}%` : "--"}
                                    </div>
                                </div>
                                <span className="text-[var(--color-nexus-text-muted)]">{expanded ? "−" : "+"}</span>
                            </button>
                            {expanded && (
                                <div className="mt-3 pt-3 border-t border-[var(--color-nexus-border)] grid grid-cols-2 gap-3 text-xs font-mono">
                                    <div className="text-[var(--color-nexus-text-secondary)]">Shares</div>
                                    <div className="text-right text-[var(--color-nexus-text-primary)]">{holding.shares != null ? holding.shares.toLocaleString() : "--"}</div>
                                    <div className="text-[var(--color-nexus-text-secondary)]">Price</div>
                                    <div className="text-right text-[var(--color-nexus-text-primary)]">{holding.price != null ? `$${holding.price.toLocaleString()}` : "--"}</div>
                                    <div className="text-[var(--color-nexus-text-secondary)]">Weight</div>
                                    <div className="text-right text-[var(--color-nexus-text-primary)]">{holding.weight != null ? `${(holding.weight * 100).toFixed(2)}%` : "--"}</div>
                                    <div className="text-[var(--color-nexus-text-secondary)]">Value</div>
                                    <div className="text-right text-[var(--color-nexus-text-primary)]">{holding.value != null ? `$${holding.value.toLocaleString()}` : "--"}</div>
                                </div>
                            )}
                        </article>
                    );
                })}
            </section>

            {holdings.length === 0 && <div className="text-center py-12 text-[var(--color-nexus-text-muted)] italic">No holdings data available</div>}
            {holdings.length > 0 && nonCashHoldings.length === 0 && <div className="text-sm text-[var(--color-nexus-text-muted)] text-center mt-4">Holdings are cash-only for this run.</div>}
        </div>
    );
}
