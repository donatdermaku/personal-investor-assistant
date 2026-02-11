"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
    { label: "Overview", href: "/overview" },
    { label: "Performance", href: "/performance" },
    { label: "Risk", href: "/risk" },
    { label: "Holdings", href: "/holdings" },
];

export function Sidebar() {
    const pathname = usePathname();

    return (
        <>
            <aside className="hidden lg:flex w-64 bg-[var(--color-nexus-surface)] border-r border-[var(--color-nexus-border)] h-screen flex-col fixed left-0 top-0 z-50">
                <div className="p-8 pb-4">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-[var(--color-nexus-primary)] shadow-[var(--shadow-glow)]" />
                        <h1 className="text-[var(--color-nexus-text-primary)] font-sans font-bold text-xl tracking-tight">NEXUS</h1>
                    </div>
                    <p className="text-[var(--color-nexus-text-muted)] text-[10px] uppercase tracking-[0.2em] mt-2 ml-4">Private Perspective</p>
                </div>

                <nav className="flex-1 px-4 py-8 space-y-2">
                    {NAV_ITEMS.map((item) => {
                        const isActive = pathname === item.href;
                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={`group flex items-center justify-between px-4 py-3 text-sm transition-all duration-300 ${isActive
                                    ? "text-[var(--color-nexus-primary)]"
                                    : "text-[var(--color-nexus-text-secondary)] hover:text-[var(--color-nexus-text-primary)]"
                                    }`}
                            >
                                <span className={`font-medium ${isActive ? "tracking-wide" : ""}`}>
                                    {item.label}
                                </span>
                                {isActive && (
                                    <div className="w-1.5 h-1.5 bg-[var(--color-nexus-primary)] rounded-full shadow-[var(--shadow-glow)]" />
                                )}
                            </Link>
                        );
                    })}
                </nav>

                <div className="p-8 border-t border-[var(--color-nexus-border)]">
                    <div className="flex items-center gap-2 text-[var(--color-nexus-text-muted)] text-xs font-mono">
                        <span className="w-2 h-2 rounded-full bg-emerald-500/20 border border-emerald-500/50" />
                        v2.0.0
                    </div>
                </div>
            </aside>

            {/* Mobile Nav */}
            <nav className="lg:hidden fixed bottom-0 left-0 right-0 bg-[var(--color-nexus-surface)] border-t border-[var(--color-nexus-border)] px-6 py-4 z-50">
                <div className="flex justify-between items-center">
                    {NAV_ITEMS.map((item) => {
                        const isActive = pathname === item.href;
                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={`text-xs uppercase tracking-wider font-medium transition-colors ${isActive ? "text-[var(--color-nexus-primary)]" : "text-[var(--color-nexus-text-secondary)]"
                                    }`}
                            >
                                {item.label}
                            </Link>
                        );
                    })}
                </div>
            </nav>
        </>
    );
}
