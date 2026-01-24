"use client";

interface ContextItem {
    label: string;
    text: string;
}

interface SectionContextProps {
    title?: string;
    items: ContextItem[];
}

export function SectionContext({ title = "Context", items }: SectionContextProps) {
    return (
        <div className="rounded-xl border border-[#E5E7EB] bg-white p-5 shadow-sm">
            <div className="inline-flex items-center rounded-full bg-[#F2F6FF] px-3 py-1 text-xs font-semibold uppercase tracking-wider text-[#1E40AF] mb-4">
                {title}
            </div>
            <div className="space-y-3 text-sm">
                {items.map((item) => (
                    <div key={item.label} className="flex gap-3">
                        <div className="min-w-[140px] text-xs uppercase tracking-wider text-gray-400">
                            {item.label}
                        </div>
                        <div className="text-gray-600">{item.text}</div>
                    </div>
                ))}
            </div>
        </div>
    );
}
