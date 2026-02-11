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
        <div className="nexus-card p-6 border-l-2 border-[var(--color-nexus-primary)]">
            <div className="text-label mb-4 text-[var(--color-nexus-primary)]">
                {title}
            </div>
            <div className="space-y-4">
                {items.map((item) => (
                    <div key={item.label} className="grid grid-cols-[120px_1fr] gap-4">
                        <div className="text-[10px] uppercase tracking-wider text-[var(--color-nexus-text-muted)] pt-0.5">
                            {item.label}
                        </div>
                        <div className="text-sm text-[var(--color-nexus-text-secondary)] font-light leading-relaxed">
                            {item.text}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
