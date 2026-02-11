"use client";

interface EmptyStateProps {
    title: string;
    description: string;
    primaryAction?: {
        label: string;
        onClick: () => void;
    };
    secondaryAction?: {
        label: string;
        onClick: () => void;
    };
}

export function EmptyState({ title, description, primaryAction, secondaryAction }: EmptyStateProps) {
    return (
        <div className="rounded-xl border border-dashed border-[#CBD5E1] bg-gradient-to-b from-white to-[#F8FAFC] p-10 text-center shadow-sm">
            <div className="mx-auto mb-4 h-10 w-10 rounded-full bg-[#DBEAFE] ring-8 ring-[#EFF6FF]" />
            <h3 className="text-lg font-semibold text-[#0F172A]">{title}</h3>
            <p className="mt-2 text-sm text-gray-600">{description}</p>
            <div className="mt-6 flex flex-wrap justify-center gap-3">
                {primaryAction && (
                    <button
                        type="button"
                        onClick={primaryAction.onClick}
                        className="rounded-md bg-[#1D4ED8] px-4 py-2 text-sm font-semibold text-white hover:bg-[#1E40AF]"
                    >
                        {primaryAction.label}
                    </button>
                )}
                {secondaryAction && (
                    <button
                        type="button"
                        onClick={secondaryAction.onClick}
                        className="rounded-md border border-[#CBD5E1] bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
                    >
                        {secondaryAction.label}
                    </button>
                )}
            </div>
        </div>
    );
}
