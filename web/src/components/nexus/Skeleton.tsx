"use client";

export function SkeletonBlock({ className }: { className?: string }) {
    return <div className={`animate-pulse rounded-md bg-gray-200 ${className || ""}`} />;
}

export function SkeletonCard() {
    return (
        <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
            <SkeletonBlock className="h-3 w-24" />
            <SkeletonBlock className="mt-4 h-6 w-20" />
            <SkeletonBlock className="mt-3 h-3 w-16" />
        </div>
    );
}
