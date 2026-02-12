"use client";

import { useEffect, useRef, useState } from "react";

interface LazyChartProps {
    heightClassName: string;
    children: React.ReactNode;
}

export function LazyChart({ heightClassName, children }: LazyChartProps) {
    const ref = useRef<HTMLDivElement | null>(null);
    const [visible, setVisible] = useState(false);

    useEffect(() => {
        const node = ref.current;
        if (!node) return;
        const observer = new IntersectionObserver(
            (entries) => {
                const entry = entries[0];
                if (entry?.isIntersecting) {
                    setVisible(true);
                    observer.disconnect();
                }
            },
            { rootMargin: "180px 0px" }
        );
        observer.observe(node);
        return () => observer.disconnect();
    }, []);

    return (
        <div ref={ref} className={heightClassName}>
            {visible ? (
                children
            ) : (
                <div className="h-full w-full bg-[var(--color-nexus-surface-hover)]/70 border border-[var(--color-nexus-border)] animate-pulse" />
            )}
        </div>
    );
}
