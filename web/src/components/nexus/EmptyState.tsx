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
        <div className="flex flex-col items-center justify-center rounded-none border border-dashed border-[var(--color-nexus-border)] bg-[var(--color-nexus-surface)]/50 p-12 text-center">
            <div className="mx-auto mb-4 h-12 w-12 rounded-full bg-[var(--color-nexus-surface-hover)] border border-[var(--color-nexus-border)] flex items-center justify-center">
                <span className="text-xl">❖</span>
            </div>
            <h3 className="text-lg font-bold text-[var(--color-nexus-text-primary)] tracking-tight">{title}</h3>
            <p className="mt-2 text-sm text-[var(--color-nexus-text-secondary)] max-w-md">{description}</p>
            <div className="mt-8 flex flex-wrap justify-center gap-4">
                {primaryAction && (
                    <button
                        type="button"
                        onClick={primaryAction.onClick}
                        className="px-6 py-2 bg-[var(--color-nexus-primary)] text-black text-xs font-mono uppercase tracking-widest font-bold hover:bg-[var(--color-nexus-accent)] transition-colors"
                    >
                        {primaryAction.label}
                    </button>
                )}
                {secondaryAction && (
                    <button
                        type="button"
                        onClick={secondaryAction.onClick}
                        className="px-6 py-2 border border-[var(--color-nexus-border)] text-[var(--color-nexus-text-secondary)] text-xs font-mono uppercase tracking-widest hover:border-[var(--color-nexus-primary)] hover:text-[var(--color-nexus-primary)] transition-colors"
                    >
                        {secondaryAction.label}
                    </button>
                )}
            </div>
        </div>
    );
}
