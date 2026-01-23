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
            <aside className="hidden lg:flex w-64 bg-white border-r border-[#E5E7EB] h-screen flex-col fixed left-0 top-0">
                <div className="p-6 border-b border-[#E5E7EB]">
                    <h1 className="text-[#0F172A] font-sans font-bold text-lg">Nexus</h1>
                    <p className="text-xs text-gray-500 uppercase tracking-wider">Analytics Platform</p>
                </div>

                <nav className="flex-1 p-4 space-y-1">
                    {NAV_ITEMS.map((item) => {
                        const isActive = pathname === item.href;
                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${isActive
                                    ? "bg-[#0F172A] text-white shadow-sm"
                                    : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                                    }`}
                            >
                                {item.label}
                            </Link>
                        );
                    })}
                </nav>

                <div className="p-4 border-t border-[#E5E7EB]">
                    <div className="text-xs text-gray-400">
                        v2.0.0 (Web Shell)
                    </div>
                </div>
            </aside>

            <nav className="lg:hidden fixed bottom-0 left-0 right-0 bg-white border-t border-[#E5E7EB] px-4 py-2">
                <div className="grid grid-cols-4 gap-2 text-center text-xs font-medium text-gray-500">
                    {NAV_ITEMS.map((item) => {
                        const isActive = pathname === item.href;
                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={isActive ? "text-[#0F172A]" : "text-gray-500"}
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
