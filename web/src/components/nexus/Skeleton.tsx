"use client";

export function SkeletonBlock({ className }: { className?: string }) {
    return <div className={`animate-pulse rounded-md bg-[var(--color-nexus-surface-hover)] ${className || ""}`} />;
}

export function SkeletonCard() {
    return (
        <div className="nexus-card">
            <SkeletonBlock className="h-3 w-24 bg-[var(--color-nexus-border)]" />
            <SkeletonBlock className="mt-4 h-6 w-20" />
            <SkeletonBlock className="mt-3 h-3 w-16 bg-[var(--color-nexus-border)]" />
        </div>
    );
}
