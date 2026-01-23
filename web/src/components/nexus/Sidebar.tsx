"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
    { label: "Overview", href: "/overview", icon: "📊" },
    { label: "Performance", href: "/performance", icon: "📈" },
    { label: "Risk", href: "/risk", icon: "⚠️" },
    { label: "Holdings", href: "/holdings", icon: "📋" },
];

export function Sidebar() {
    const pathname = usePathname();

    return (
        <aside className="w-64 bg-white border-r border-[#E5E7EB] h-screen flex flex-col fixed left-0 top-0">
            <div className="p-6 border-b border-[#E5E7EB]">
                <h1 className="text-[#0F172A] font-sans font-bold text-lg">Nexus Analytics</h1>
                <p className="text-xs text-gray-500 uppercase tracking-wider">Platform</p>
            </div>

            <nav className="flex-1 p-4 space-y-1">
                {NAV_ITEMS.map((item) => {
                    const isActive = pathname === item.href;
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${isActive
                                    ? "bg-[#F3F4F6] text-[#0F172A]"
                                    : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                                }`}
                        >
                            <span>{item.icon}</span>
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
    );
}
