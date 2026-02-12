"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useNexus } from "@/components/nexus/NexusProvider";

const NAV_ITEMS = [
    { label: "Overview", href: "/overview" },
    { label: "Performance", href: "/performance" },
    { label: "Risk", href: "/risk" },
    { label: "Holdings", href: "/holdings" },
];

export function Sidebar() {
    const pathname = usePathname();
    const { navOpen, closeNav } = useNexus();

    const navLinks = (onNavigate?: () => void) => (
        <>
            {NAV_ITEMS.map((item) => {
                const isActive = pathname === item.href;
                return (
                    <Link
                        key={item.href}
                        href={item.href}
                        onClick={onNavigate}
                        className={`group flex items-center justify-between px-4 py-3 text-sm transition-all duration-300 nexus-touch-target ${isActive
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
        </>
    );

    return (
        <>
            <aside className="hidden lg:flex flex-col bg-[var(--color-nexus-surface)]/90 border-r border-[var(--color-nexus-border)] min-h-screen sticky top-0">
                <div className="p-6 xl:p-8 pb-4">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-[var(--color-nexus-primary)] shadow-[var(--shadow-glow)]" />
                        <h1 className="text-[var(--color-nexus-text-primary)] font-sans font-bold text-xl tracking-tight">NEXUS</h1>
                    </div>
                    <p className="text-[var(--color-nexus-text-muted)] text-[10px] uppercase tracking-[0.2em] mt-2 ml-4">Private Perspective</p>
                </div>

                <nav className="flex-1 px-4 py-8 space-y-2">
                    {navLinks()}
                </nav>

                <div className="p-6 xl:p-8 border-t border-[var(--color-nexus-border)]">
                    <div className="flex items-center gap-2 text-[var(--color-nexus-text-muted)] text-xs font-mono">
                        <span className="w-2 h-2 rounded-full bg-emerald-500/20 border border-emerald-500/50" />
                        v2.0.0
                    </div>
                </div>
            </aside>

            <div
                className={`fixed inset-0 z-40 bg-black/50 backdrop-blur-sm transition-opacity duration-300 lg:hidden ${navOpen ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"}`}
                onClick={closeNav}
            />
            <aside
                className={`fixed inset-y-0 left-0 w-[280px] max-w-[85vw] z-50 bg-[var(--color-nexus-surface)] border-r border-[var(--color-nexus-border)] p-6 flex flex-col transform transition-transform duration-300 lg:hidden ${navOpen ? "translate-x-0" : "-translate-x-full"}`}
            >
                <div className="flex items-center justify-between pb-4 border-b border-[var(--color-nexus-border)]">
                    <h2 className="text-sm uppercase tracking-widest text-[var(--color-nexus-text-secondary)]">Navigation</h2>
                    <button
                        type="button"
                        onClick={closeNav}
                        className="nexus-touch-target text-[var(--color-nexus-text-secondary)] hover:text-[var(--color-nexus-primary)]"
                    >
                        Close
                    </button>
                </div>
                <nav className="flex-1 pt-6 space-y-2">
                    {navLinks(closeNav)}
                </nav>
            </aside>

            <nav className="fixed bottom-0 left-0 right-0 bg-[var(--color-nexus-surface)] border-t border-[var(--color-nexus-border)] px-4 py-3 z-50 md:hidden">
                <div className="flex justify-between items-center">
                    {NAV_ITEMS.map((item) => {
                        const isActive = pathname === item.href;
                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={`text-xs uppercase tracking-wider font-medium transition-colors px-2 py-1.5 nexus-touch-target inline-flex items-center justify-center ${isActive ? "text-[var(--color-nexus-primary)]" : "text-[var(--color-nexus-text-secondary)]"
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
