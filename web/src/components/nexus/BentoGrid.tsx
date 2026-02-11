import React from "react";

interface BentoGridProps {
    children: React.ReactNode;
    className?: string;
}

interface BentoItemProps {
    children: React.ReactNode;
    className?: string;
    span?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12; // Span across columns (12-grid system)
    rowSpan?: number;
    title?: string;
    action?: React.ReactNode;
}

export function BentoGrid({ children, className = "" }: BentoGridProps) {
    return (
        <div className={`grid grid-cols-1 md:grid-cols-12 gap-6 auto-rows-min md:auto-rows-[12rem] ${className}`}>
            {children}
        </div>
    );
}

export function BentoItem({
    children,
    className = "",
    span = 4,
    rowSpan = 1,
    title,
    action
}: BentoItemProps) {
    // Calculate column span classes
    const spanClass = {
        1: "md:col-span-1",
        2: "md:col-span-2",
        3: "md:col-span-3",
        4: "md:col-span-4",
        5: "md:col-span-5",
        6: "md:col-span-6",
        7: "md:col-span-7",
        8: "md:col-span-8",
        9: "md:col-span-9",
        10: "md:col-span-10",
        11: "md:col-span-11",
        12: "md:col-span-12",
    }[span];

    const rowSpanClass = rowSpan > 1 ? `row-span-${rowSpan}` : "";

    return (
        <div
            className={`
                nexus-card relative group flex flex-col overflow-hidden
                ${spanClass} ${rowSpanClass} ${className}
                transition-all duration-500
                border border-[var(--color-nexus-border)]/50
                backdrop-blur-md bg-[var(--color-nexus-surface)]/80
                hover:border-[var(--color-nexus-primary)]/50
                hover:shadow-[0_0_30px_rgba(212,175,55,0.1)]
                hover:-translate-y-1
            `}
        >
            {/* Soul Glow Gradient */}
            <div className="absolute inset-0 bg-gradient-to-br from-[var(--color-nexus-primary)]/10 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none" />

            {(title || action) && (
                <div className="flex items-center justify-between mb-4 p-6 pb-0 relative z-10">
                    {title && (
                        <h3 className="text-lg font-bold text-[var(--color-nexus-text-primary)] tracking-tight">
                            {title}
                        </h3>
                    )}
                    {action && (
                        <div className="text-xs">
                            {action}
                        </div>
                    )}
                </div>
            )}

            <div className="p-6 relative z-10 flex-1 flex flex-col">
                {children}
            </div>
        </div>
    );
}
