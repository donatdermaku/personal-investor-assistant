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
    // Don't hide if coverage is "available_low_coverage", only if "insufficient"
    const showValue = shouldHide ? "--" : safeValue;
    const asOfLabel = formatAsOf(asOf);
    const badgeLabel = coverageStatus === "full"
        ? "Full coverage"
        : coverageStatus === "partial"
            ? "Partial coverage"
            : coverageStatus === "insufficient"
                ? "Insufficient"
                : coverageStatus === "available_low_coverage"
                    ? "Warning"
                    : null;
    const badgeClass = coverageStatus === "full"
        ? "border-emerald-200 bg-emerald-50 text-emerald-700"
        : coverageStatus === "partial"
            ? "border-amber-200 bg-amber-50 text-amber-700"
            : coverageStatus === "insufficient"
                ? "border-gray-200 bg-gray-100 text-gray-500"
                : coverageStatus === "available_low_coverage"
                    ? "border-yellow-200 bg-yellow-50 text-yellow-700"
                    : "border-[#E8F0FF] bg-[#E8F0FF] text-[#1E40AF]";

    const reasonLabel =
        shouldHide && reasonCodes && reasonCodes.length > 0
            ? `Reason: ${reasonCodes.join(", ")}`
            : null;

    return (
        <div
            className={`nexus-card border-l-4 border-[#E8F0FF] transition-all ${coverageStatus === "insufficient" ? "opacity-70" : "hover:border-[#2563EB] hover:shadow-md"
                }`}
            title={tooltip}
        >
            <div className="flex items-center justify-between">
                <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
                    {label}
                </div>
                {badgeLabel && (
                    <div className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${badgeClass}`}>
                        {badgeLabel}
                    </div>
                )}
            </div>
            <div className="text-2xl font-bold text-[#0F172A]">
                {showValue}
            </div>
            {(subtext || asOfLabel || reasonLabel) && (
                <div className="text-xs text-gray-400 mt-1">
                    {subtext}
                    {subtext && asOfLabel ? " · " : ""}
                    {asOfLabel ? `As of ${asOfLabel}` : ""}
                    {(subtext || asOfLabel) && reasonLabel ? " · " : ""}
                    {reasonLabel ?? ""}
                </div>
            )}
        </div>
    );
}
