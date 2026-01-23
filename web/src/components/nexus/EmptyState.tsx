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
        <div className="rounded-xl border border-dashed border-gray-300 bg-white p-10 text-center shadow-sm">
            <h3 className="text-lg font-semibold text-[#0F172A]">{title}</h3>
            <p className="mt-2 text-sm text-gray-500">{description}</p>
            <div className="mt-6 flex flex-wrap justify-center gap-3">
                {primaryAction && (
                    <button
                        type="button"
                        onClick={primaryAction.onClick}
                        className="rounded-md bg-[#0F172A] px-4 py-2 text-sm font-semibold text-white"
                    >
                        {primaryAction.label}
                    </button>
                )}
                {secondaryAction && (
                    <button
                        type="button"
                        onClick={secondaryAction.onClick}
                        className="rounded-md border border-gray-200 px-4 py-2 text-sm font-semibold text-gray-700"
                    >
                        {secondaryAction.label}
                    </button>
                )}
            </div>
        </div>
    );
}
