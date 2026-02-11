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

export function RunCreationModal() {
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

    useEffect(() => {
        if (!runCreatorOpen) {
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
    }, [runCreatorOpen]);

    if (!runCreatorOpen) return null;

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
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 px-4">
            <div className="w-full max-w-xl rounded-xl bg-white p-6 shadow-xl">
                <div className="flex items-center justify-between">
                    <div>
                        <h3 className="text-lg font-semibold text-[#0F172A]">Create a Portfolio Run</h3>
                        <p className="text-sm text-gray-500">
                            Portfolio: <span className="font-semibold text-gray-700">{portfolioId}</span>
                        </p>
                    </div>
                    <button
                        type="button"
                        onClick={closeRunCreator}
                        className="text-sm text-gray-500 hover:text-gray-700"
                    >
                        Close
                    </button>
                </div>

                <div className="mt-6 space-y-4">
                    <div className="rounded-lg border border-[#E3E7EE] bg-[#F8FAFC] p-4">
                        <div className="text-xs font-semibold uppercase tracking-wider text-[#1E40AF]">
                            Upload Trades CSV
                        </div>
                        <p className="mt-2 text-sm text-gray-600">
                            Required columns: date, ticker, action, quantity (or shares), price. Optional: fees.
                        </p>
                        <input
                            type="file"
                            accept=".csv"
                            onChange={handleFileChange}
                            className="mt-3 block w-full text-sm text-gray-600 file:mr-3 file:rounded-md file:border-0 file:bg-[#E8F0FF] file:px-3 file:py-1.5 file:text-sm file:font-semibold file:text-[#1E40AF]"
                        />
                        {fileError && (
                            <div className="mt-2 text-xs text-red-600">{fileError}</div>
                        )}
                        <button
                            type="button"
                            disabled={isSubmitting || !!fileError}
                            onClick={handleUpload}
                            className="mt-4 rounded-md bg-[#2563EB] px-4 py-2 text-sm font-semibold text-white disabled:opacity-60"
                        >
                            {isSubmitting ? "Creating run..." : "Upload & Compute"}
                        </button>
                    </div>

                    <div className="rounded-lg border border-[#E3E7EE] p-4">
                        <div className="text-xs font-semibold uppercase tracking-wider text-[#1E40AF]">
                            Create Demo Run
                        </div>
                        <p className="mt-2 text-sm text-gray-600">
                            Generate a demo portfolio run using your watchlist data.
                        </p>
                        <button
                            type="button"
                            disabled={isSubmitting}
                            onClick={handleCreateDemo}
                            className="mt-3 rounded-md border border-[#2563EB] px-4 py-2 text-sm font-semibold text-[#1E40AF] hover:bg-[#E8F0FF] disabled:opacity-60"
                        >
                            {isSubmitting ? "Creating run..." : "Create Demo Run"}
                        </button>
                    </div>
                </div>

                {isSubmitting && (
                    <div className="mt-4 rounded-md border border-[#DBEAFE] bg-[#EFF6FF] px-3 py-3 text-sm text-[#1E40AF]">
                        <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wide">
                            <span>{submitMode === "demo" ? "Demo Run Progress" : "Upload Progress"}</span>
                            <span>{Math.max(5, Math.min(100, Math.round(submitProgress)))}%</span>
                        </div>
                        <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-[#DBEAFE]">
                            <div
                                className="h-full rounded-full bg-[#2563EB] transition-all duration-300"
                                style={{ width: `${Math.max(5, Math.min(100, Math.round(submitProgress)))}%` }}
                            />
                        </div>
                        <div className="mt-2 text-xs text-[#1D4ED8]">{submitStep || "Starting..."}</div>
                    </div>
                )}

                {submitWarning && (
                    <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
                        {submitWarning}
                    </div>
                )}

                {submitError && (
                    <div className="mt-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                        {submitError}
                        {submitHint && <div className="mt-1 text-xs text-red-600">{submitHint}</div>}
                        <div className="mt-1 text-xs text-red-600">
                            Common fixes: validate CSV headers, ensure dates are ISO format, and retry with a smaller file.
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
