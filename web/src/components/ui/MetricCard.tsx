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
            className="nexus-card hover:shadow-md transition-shadow"
            title={tooltip}
        >
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
                {label}
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
