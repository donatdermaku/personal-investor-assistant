"use client";

import { useEffect, useState } from "react";
import type { ChangeEvent } from "react";
import { useRouter } from "next/navigation";
import { useNexus } from "@/components/nexus/NexusProvider";

const REQUIRED_COLUMNS = ["date", "ticker", "action", "price"];

function parseHeaders(text: string): string[] {
    const [headerLine] = text.split(/\r?\n/);
    if (!headerLine) return [];
    return headerLine
        .split(",")
        .map((value) => value.replace(/^"|"$/g, "").trim().toLowerCase())
        .filter(Boolean);
}

interface RunCreationModalProps {
    isOpen?: boolean;
    onClose?: () => void;
}

export function RunCreationModal({ isOpen, onClose }: RunCreationModalProps) {
    const {
        runCreatorOpen,
        closeRunCreator,
        createRun,
        portfolioId,
    } = useNexus();
    const router = useRouter();
    const [file, setFile] = useState<File | null>(null);
    const [fileError, setFileError] = useState<string | null>(null);
    const [submitError, setSubmitError] = useState<string | null>(null);
    const [submitHint, setSubmitHint] = useState<string | null>(null);
    const [submitWarning, setSubmitWarning] = useState<string | null>(null);
    const [submitProgress, setSubmitProgress] = useState(0);
    const [submitStep, setSubmitStep] = useState<string | null>(null);
    const [submitMode, setSubmitMode] = useState<"upload" | "demo" | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Use passed props if available (from MissionControl), otherwise fall back to context (global trigger)
    // If isOpen is provided, use it. If not, use runCreatorOpen from context.
    // However, if isOpen IS provided, we should probably ignore context or OR them?
    // The requirement is MissionControl controls its own modal state.
    // Let's say: if isOpen is defined, we use it. If undefined, we use context.
    const show = isOpen !== undefined ? isOpen : runCreatorOpen;
    const handleClose = onClose || closeRunCreator;

    // We need to keep the useEffect for resetting state when the modal closes
    useEffect(() => {
        if (!show) {
            setFile(null);
            setFileError(null);
            setSubmitError(null);
            setSubmitHint(null);
            setSubmitWarning(null);
            setSubmitProgress(0);
            setSubmitStep(null);
            setSubmitMode(null);
            setIsSubmitting(false);
        }
    }, [show]);

    if (!show) return null;

    const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
        const selected = event.target.files?.[0] ?? null;
        setFile(selected);
        setFileError(null);
        setSubmitError(null);
        setSubmitWarning(null);
        if (!selected) return;

        try {
            const text = await selected.text();
            const headers = parseHeaders(text);
            const missing = REQUIRED_COLUMNS.filter((col) => !headers.includes(col));
            const hasQuantity = headers.includes("quantity") || headers.includes("shares");
            if (!hasQuantity) {
                missing.push("quantity (or shares)");
            }
            if (missing.length > 0) {
                setFileError(`Missing columns: ${missing.join(", ")}`);
            }
            if (selected.size > 4 * 1024 * 1024) {
                setFileError("CSV is large (>4MB). Upload may take longer. Consider reducing file size.");
            }
        } catch (err) {
            setFileError(err instanceof Error ? err.message : "Unable to read CSV file.");
        }
    };

    const startSubmitProgress = (initialStep: string) => {
        setSubmitProgress(12);
        setSubmitStep(initialStep);
        return window.setInterval(() => {
            setSubmitProgress((prev) => {
                if (prev >= 90) return prev;
                return prev + 8;
            });
        }, 350);
    };

    const handleCreateDemo = async () => {
        setSubmitError(null);
        setSubmitHint(null);
        setSubmitWarning(null);
        setSubmitMode("demo");
        const intervalId = startSubmitProgress("Preparing demo portfolio");
        setIsSubmitting(true);
        try {
            setSubmitStep("Computing demo analytics");
            const result = await createRun({ runType: "demo" });
            if (result.warnings?.failed_tickers?.message) {
                setSubmitWarning(result.warnings.failed_tickers.message);
            }
            setSubmitProgress(100);
            setSubmitStep("Demo run ready");
            router.push("/overview");
        } catch (err) {
            setSubmitError(err instanceof Error ? err.message : "Failed to create demo run.");
            setSubmitHint("Verify backend health and market data availability, then retry.");
        } finally {
            window.clearInterval(intervalId);
            setIsSubmitting(false);
        }
    };

    const handleUpload = async () => {
        if (!file) {
            setFileError("Please select a CSV file.");
            return;
        }
        if (fileError) return;
        setSubmitError(null);
        setSubmitHint(null);
        setSubmitWarning(null);
        setSubmitMode("upload");
        const intervalId = startSubmitProgress("Validating CSV format");
        setIsSubmitting(true);
        try {
            setSubmitStep("Uploading trades and fetching market data");
            const result = await createRun({ runType: "uploaded", file });
            if (result.warnings?.failed_tickers?.message) {
                setSubmitWarning(result.warnings.failed_tickers.message);
            }
            setSubmitProgress(100);
            setSubmitStep("Run completed");
            router.push("/overview");
        } catch (err) {
            setSubmitError(err instanceof Error ? err.message : "Run creation failed.");
            setSubmitHint("Check CSV columns, date values, and market data coverage before retrying.");
        } finally {
            window.clearInterval(intervalId);
            setIsSubmitting(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm px-4">
            <div className="w-full max-w-xl nexus-card p-0 overflow-hidden shadow-2xl skew-y-0">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-[var(--color-nexus-border)]">
                    <h2 className="text-xl font-sans font-bold text-[var(--color-nexus-text-primary)] tracking-tight">
                        Create New Run
                    </h2>
                    <button
                        onClick={handleClose}
                        className="text-[var(--color-nexus-text-secondary)] hover:text-[var(--color-nexus-primary)] transition-colors"
                    >
                        ✕
                    </button>
                </div>

                <div className="p-6 space-y-6 bg-[var(--color-nexus-surface)]/80 backdrop-blur-md">
                    {/* CSV Upload Section */}
                    <div className="rounded-none border border-[var(--color-nexus-border)] bg-[var(--color-nexus-surface-hover)] p-4 relative overflow-hidden group">
                        <div className="absolute top-0 left-0 w-1 h-full bg-[var(--color-nexus-primary)]" />
                        <div className="text-xs font-mono font-bold uppercase tracking-widest text-[var(--color-nexus-primary)] mb-2">
                            Option A: Upload Trades
                        </div>
                        <p className="text-xs text-[var(--color-nexus-text-secondary)] mb-4">
                            Upload a CSV with columns: <span className="font-mono text-[var(--color-nexus-text-primary)]">date, ticker, action, quantity, price</span>.
                        </p>

                        <div className="relative">
                            <input
                                type="file"
                                accept=".csv"
                                onChange={handleFileChange}
                                className="block w-full text-xs text-[var(--color-nexus-text-secondary)] 
                                    file:mr-4 file:py-2 file:px-4
                                    file:rounded-none file:border-0
                                    file:text-xs file:font-mono file:uppercase file:font-bold
                                    file:bg-[var(--color-nexus-primary)] file:text-black
                                    hover:file:bg-[var(--color-nexus-accent)]
                                    cursor-pointer"
                            />
                        </div>

                        {fileError && (
                            <div className="mt-3 text-xs text-[var(--color-nexus-danger)] font-mono border-l-2 border-[var(--color-nexus-danger)] pl-2">
                                {fileError}
                            </div>
                        )}

                        <button
                            type="button"
                            disabled={isSubmitting || !!fileError}
                            onClick={handleUpload}
                            className="mt-4 w-full py-2 bg-[var(--color-nexus-surface)] border border-[var(--color-nexus-border)] text-[var(--color-nexus-text-primary)] hover:border-[var(--color-nexus-primary)] hover:text-[var(--color-nexus-primary)] text-xs font-mono uppercase tracking-widest transition-all disabled:opacity-50 disabled:hover:border-[var(--color-nexus-border)]"
                        >
                            {isSubmitting && submitMode === "upload" ? "Computing..." : "Start Computation"}
                        </button>
                    </div>

                    {/* Divider */}
                    <div className="flex items-center gap-4">
                        <div className="h-px flex-1 bg-[var(--color-nexus-border)]" />
                        <span className="text-[10px] text-[var(--color-nexus-text-muted)] font-mono uppercase">OR</span>
                        <div className="h-px flex-1 bg-[var(--color-nexus-border)]" />
                    </div>

                    {/* Demo Run Section */}
                    <div className="rounded-none border border-[var(--color-nexus-border)] p-4 hover:border-[var(--color-nexus-text-muted)] transition-colors">
                        <div className="flex items-center justify-between">
                            <div>
                                <div className="text-xs font-mono font-bold uppercase tracking-widest text-[var(--color-nexus-text-secondary)]">
                                    Option B: Demo Simulation
                                </div>
                                <p className="text-xs text-[var(--color-nexus-text-muted)] mt-1">
                                    Generate analytics using sample market data.
                                </p>
                            </div>
                            <button
                                type="button"
                                disabled={isSubmitting}
                                onClick={handleCreateDemo}
                                className="px-4 py-2 border border-[var(--color-nexus-border)] text-[var(--color-nexus-text-secondary)] hover:bg-[var(--color-nexus-surface-hover)] hover:text-[var(--color-nexus-primary)] text-xs font-mono uppercase tracking-widest transition-all disabled:opacity-50"
                            >
                                {isSubmitting && submitMode === "demo" ? "Loading..." : "Load Demo"}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Progress Bar */}
                {isSubmitting && (
                    <div className="p-6 pt-0 bg-[var(--color-nexus-surface)]/80 backdrop-blur-md">
                        <div className="border border-[var(--color-nexus-primary)]/20 bg-[var(--color-nexus-primary)]/5 p-4 relative overflow-hidden">
                            <div className="flex items-center justify-between text-[10px] font-mono uppercase tracking-widest text-[var(--color-nexus-primary)] mb-2 relative z-10">
                                <span>{submitMode === "demo" ? "Simulation Progress" : "Processing Pipeline"}</span>
                                <span>{Math.round(submitProgress)}%</span>
                            </div>
                            <div className="h-1 w-full bg-[var(--color-nexus-border)] overflow-hidden relative z-10">
                                <div
                                    className="h-full bg-[var(--color-nexus-primary)] shadow-[0_0_10px_var(--color-nexus-primary)] transition-all duration-300"
                                    style={{ width: `${submitProgress}%` }}
                                />
                            </div>
                            <div className="mt-2 text-[10px] text-[var(--color-nexus-text-secondary)] font-mono relative z-10">
                                {submitStep || "Initializing..."}
                            </div>
                            {/* Animated sheen */}
                            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-[var(--color-nexus-primary)]/5 to-transparent animate-pulse-slow pointer-events-none" />
                        </div>
                    </div>
                )}

                {/* Status Messages */}
                {(submitError || submitWarning) && (
                    <div className="p-6 pt-0 bg-[var(--color-nexus-surface)]/80 backdrop-blur-md">
                        {submitWarning && (
                            <div className="p-3 border border-[var(--color-nexus-warning)] bg-[var(--color-nexus-warning)]/10 text-xs text-[var(--color-nexus-warning)] font-mono">
                                <span className="font-bold">WARNING:</span> {submitWarning}
                            </div>
                        )}
                        {submitError && (
                            <div className="p-3 border border-[var(--color-nexus-danger)] bg-[var(--color-nexus-danger)]/10 text-xs text-[var(--color-nexus-danger)] font-mono">
                                <div className="font-bold mb-1">EXECUTION FAILED</div>
                                {submitError}
                                {submitHint && <div className="mt-1 opacity-75">{submitHint}</div>}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
