"use client";

import type { CoverageStatus } from "@/lib/coverage";

interface MetricCardProps {
    label: string;
    value: string | number | null;
    subtext?: string;
    tooltip?: string;
    asOf?: string | null;
    coverageStatus?: CoverageStatus;
    hideWhenInsufficient?: boolean;
    reasonCodes?: string[];
}

function formatAsOf(value?: string | null) {
    if (!value) return null;
    if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
        return value;
    }
    if (value.includes("T")) {
        return value.replace("Z", "").split(".")[0].replace("T", " ");
    }
    return value;
}

export function MetricCard({
    label,
    value,
    subtext,
    tooltip,
    asOf,
    coverageStatus,
    hideWhenInsufficient,
    reasonCodes,
}: MetricCardProps) {
    const shouldHide = coverageStatus === "insufficient";
    if (shouldHide && hideWhenInsufficient) {
        return null;
    }

    const safeValue = value === null || value === undefined ? "--" : value;
    const showValue = shouldHide ? "--" : safeValue;
    const asOfLabel = formatAsOf(asOf);

    // Badge logic adapted for Dark Mode
    const badgeLabel = coverageStatus === "full"
        ? "Full Coverage"
        : coverageStatus === "partial"
            ? "Partial"
            : coverageStatus === "insufficient"
                ? "Insufficient"
                : coverageStatus === "available_low_coverage"
                    ? "Warning"
                    : null;

    const badgeClass = coverageStatus === "full"
        ? "border-[var(--color-nexus-success)] text-[var(--color-nexus-success)] bg-[var(--color-nexus-success)]/10"
        : coverageStatus === "partial"
            ? "border-[var(--color-nexus-warning)] text-[var(--color-nexus-warning)] bg-[var(--color-nexus-warning)]/10"
            : coverageStatus === "insufficient"
                ? "border-[var(--color-nexus-border)] text-[var(--color-nexus-text-muted)] bg-[var(--color-nexus-surface-hover)]"
                : coverageStatus === "available_low_coverage"
                    ? "border-[var(--color-nexus-warning)] text-[var(--color-nexus-warning)]"
                    : "border-[var(--color-nexus-primary)] text-[var(--color-nexus-primary)] bg-[var(--color-nexus-primary)]/10";

    const reasonLabel =
        shouldHide && reasonCodes && reasonCodes.length > 0
            ? `Reason: ${reasonCodes.join(", ")}`
            : null;

    return (
        <div
            className={`nexus-card relative overflow-hidden transition-all group ${coverageStatus === "insufficient" ? "opacity-50" : "hover:border-[var(--color-nexus-primary)]"}`}
            title={tooltip}
        >
            {/* Hover Glow Effect */}
            <div className="absolute -inset-px bg-gradient-to-r from-[var(--color-nexus-primary)] to-[var(--color-nexus-accent)] opacity-0 group-hover:opacity-10 pointer-events-none transition-opacity duration-500" />

            <div className="relative p-6">
                <div className="flex items-center justify-between mb-4">
                    <div className="text-label">
                        {label}
                    </div>
                    {badgeLabel && (
                        <div className={`rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wider font-mono ${badgeClass}`}>
                            {badgeLabel}
                        </div>
                    )}
                </div>

                <div className="text-3xl font-mono font-bold text-[var(--color-nexus-text-primary)] tracking-tight">
                    {showValue}
                </div>

                {(subtext || asOfLabel || reasonLabel) && (
                    <div className="text-xs font-mono text-[var(--color-nexus-text-secondary)] mt-2 flex flex-wrap gap-1">
                        {subtext && <span>{subtext}</span>}
                        {subtext && asOfLabel && <span className="opacity-50">·</span>}
                        {asOfLabel && <span>{asOfLabel}</span>}
                        {(subtext || asOfLabel) && reasonLabel && <span className="opacity-50">·</span>}
                        {reasonLabel && <span className="text-[var(--color-nexus-danger)]">{reasonLabel}</span>}
                    </div>
                )}
            </div>
        </div>
    );
}
