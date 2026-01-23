"use client";

interface MetricCardProps {
    label: string;
    value: string | number;
    subtext?: string;
    tooltip?: string;
}

export function MetricCard({ label, value, subtext, tooltip }: MetricCardProps) {
    return (
        <div
            className="nexus-card border-l-4 border-[#E8F0FF] hover:border-[#2563EB] hover:shadow-md transition-all"
            title={tooltip}
        >
            <div className="flex items-center justify-between">
                <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
                    {label}
                </div>
                <div className="rounded-full bg-[#E8F0FF] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-[#1E40AF]">
                    KPI
                </div>
            </div>
            <div className="text-2xl font-bold text-[#0F172A]">
                {value}
            </div>
            {subtext && (
                <div className="text-xs text-gray-400 mt-1">
                    {subtext}
                </div>
            )}
        </div>
    );
}
