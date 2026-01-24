"use client";

interface MetricCardProps {
    label: string;
    value: string | number | null;
    subtext?: string;
    tooltip?: string;
    asOf?: string | null;
    coverageStatus?: "full" | "partial" | "insufficient";
    hideWhenInsufficient?: boolean;
}

function formatAsOf(value?: string | null) {
    if (!value) return null;
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleDateString();
}

export function MetricCard({
    label,
    value,
    subtext,
    tooltip,
    asOf,
    coverageStatus,
    hideWhenInsufficient,
}: MetricCardProps) {
    if (coverageStatus === "insufficient" && hideWhenInsufficient) {
        return null;
    }

    const safeValue = value === null || value === undefined ? "--" : value;
    const showValue = coverageStatus === "insufficient" ? "--" : safeValue;
    const asOfLabel = formatAsOf(asOf);
    const badgeLabel = coverageStatus === "full"
        ? "Full coverage"
        : coverageStatus === "partial"
            ? "Partial coverage"
            : coverageStatus === "insufficient"
                ? "Insufficient"
                : null;
    const badgeClass = coverageStatus === "full"
        ? "border-emerald-200 bg-emerald-50 text-emerald-700"
        : coverageStatus === "partial"
            ? "border-amber-200 bg-amber-50 text-amber-700"
            : coverageStatus === "insufficient"
                ? "border-gray-200 bg-gray-100 text-gray-500"
                : "border-[#E8F0FF] bg-[#E8F0FF] text-[#1E40AF]";

    return (
        <div
            className={`nexus-card border-l-4 border-[#E8F0FF] transition-all ${
                coverageStatus === "insufficient" ? "opacity-70" : "hover:border-[#2563EB] hover:shadow-md"
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
            {(subtext || asOfLabel) && (
                <div className="text-xs text-gray-400 mt-1">
                    {subtext}
                    {subtext && asOfLabel ? " · " : ""}
                    {asOfLabel ? `As of ${asOfLabel}` : ""}
                </div>
            )}
        </div>
    );
}
