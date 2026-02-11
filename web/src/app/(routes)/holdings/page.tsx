"use client";

import { useNexus } from "@/components/nexus/NexusProvider";
import { EmptyState } from "@/components/nexus/EmptyState";
import { SkeletonBlock } from "@/components/nexus/Skeleton";
import { SectionContext } from "@/components/nexus/SectionContext";
import { coverageLabel, coverageStatus } from "@/lib/coverage";

export default function HoldingsPage() {
    const { state, status, error, mode, setMode, openRunCreator } = useNexus();

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
            <div className="space-y-6">
                <SkeletonBlock className="h-8 w-40 bg-[var(--color-nexus-surface)]" />
                <SkeletonBlock className="h-4 w-56 bg-[var(--color-nexus-surface)]" />
                <div className="nexus-card p-6 min-h-[400px]">
                    <SkeletonBlock className="h-4 w-32 bg-[var(--color-nexus-surface-hover)]" />
                    <SkeletonBlock className="mt-4 h-full w-full bg-[var(--color-nexus-surface-hover)]" />
                </div>
            </div>
        );
    }

    const { holdings, summary, manifest, coverage_summary } = state;
    const coverage = coverageStatus(manifest, coverage_summary ?? null);
    const coverageText = coverageLabel(coverage);
    const asOf = summary.last_date || manifest.timestamp;
    const nonCashHoldings = holdings.filter((holding) => holding.ticker !== "CASH");

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            <div>
                <h2 className="text-3xl font-sans font-bold text-[var(--color-nexus-text-primary)] mb-2 tracking-tight">Holdings</h2>
                <p className="text-[var(--color-nexus-text-secondary)]">
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

            {/* Holdings Table */}
            <div className="nexus-card overflow-hidden">
                <table className="min-w-full divide-y divide-[var(--color-nexus-border)] text-sm">
                    <thead className="bg-[var(--color-nexus-surface-hover)]">
                        <tr>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-[var(--color-nexus-text-secondary)] uppercase tracking-wider font-mono">
                                Ticker
                            </th>
                            <th className="px-6 py-3 text-right text-xs font-semibold text-[var(--color-nexus-text-secondary)] uppercase tracking-wider font-mono">
                                Shares
                            </th>
                            <th className="px-6 py-3 text-right text-xs font-semibold text-[var(--color-nexus-text-secondary)] uppercase tracking-wider font-mono">
                                Price
                            </th>
                            <th className="px-6 py-3 text-right text-xs font-semibold text-[var(--color-nexus-text-secondary)] uppercase tracking-wider font-mono">
                                Weight
                            </th>
                            <th className="px-6 py-3 text-right text-xs font-semibold text-[var(--color-nexus-text-secondary)] uppercase tracking-wider font-mono">
                                Value
                            </th>
                        </tr>
                    </thead>
                    <tbody className="bg-[var(--color-nexus-surface)] divide-y divide-[var(--color-nexus-border)]">
                        {holdings.map((holding, idx) => (
                            <tr key={idx} className="hover:bg-[var(--color-nexus-surface-hover)] transition-colors group">
                                <td className="px-6 py-4 whitespace-nowrap">
                                    <div className="text-sm font-medium text-[var(--color-nexus-text-primary)] font-mono group-hover:text-[var(--color-nexus-primary)] transition-colors">{holding.ticker}</div>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-right">
                                    <div className="text-sm text-[var(--color-nexus-text-secondary)] font-mono">
                                        {holding.shares != null ? holding.shares.toLocaleString() : "--"}
                                    </div>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-right">
                                    <div className="text-sm text-[var(--color-nexus-text-secondary)] font-mono">
                                        {holding.price != null ? `$${holding.price.toLocaleString()}` : "--"}
                                    </div>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-right relative">
                                    {/* Weight Bar Background */}
                                    {holding.weight != null && (
                                        <div
                                            className="absolute inset-y-2 right-2 bg-[var(--color-nexus-primary)]/10 rounded-l"
                                            style={{
                                                width: `${Math.min(holding.weight * 100 * 2, 100)}px`, // Visual scaling, max 100px width 
                                                opacity: 0.5
                                            }}
                                        />
                                    )}
                                    <div className="relative z-10 text-sm text-[var(--color-nexus-text-primary)] font-mono font-bold">
                                        {holding.weight != null ? `${(holding.weight * 100).toFixed(1)}%` : "--"}
                                    </div>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-right">
                                    <div className="text-sm font-medium text-[var(--color-nexus-text-primary)] font-mono">
                                        {holding.value != null ? `$${holding.value.toLocaleString()}` : "--"}
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {holdings.length === 0 && (
                <div className="text-center py-12 text-[var(--color-nexus-text-muted)] italic">
                    No holdings data available
                </div>
            )}

            {holdings.length > 0 && nonCashHoldings.length === 0 && (
                <div className="text-sm text-[var(--color-nexus-text-muted)] text-center mt-4">
                    Holdings are cash-only for this run.
                </div>
            )}
        </div>
    );
}
