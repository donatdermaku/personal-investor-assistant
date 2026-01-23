"use client";

import { useNexus } from "@/components/nexus/NexusProvider";

function formatTimestamp(value: string | null) {
    if (!value) return "--";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleTimeString();
}

export function TopBar() {
    const {
        mode,
        setMode,
        portfolioId,
        setPortfolioId,
        benchmark,
        setBenchmark,
        status,
        lastFetched,
        backendOk,
        refresh,
    } = useNexus();

    return (
        <div className="mb-8 flex flex-col gap-4 rounded-xl border border-gray-200 bg-white p-4 shadow-sm lg:flex-row lg:items-center lg:justify-between">
            <div className="flex flex-wrap items-center gap-3">
                <div className="text-xs uppercase tracking-wider text-gray-400">Mode</div>
                <div className="flex items-center rounded-full border border-gray-200 bg-gray-50 p-1 text-xs font-semibold">
                    <button
                        type="button"
                        onClick={() => setMode("live")}
                        className={`px-3 py-1 rounded-full ${mode === "live" ? "bg-[#0F172A] text-white" : "text-gray-500"}`}
                    >
                        Live
                    </button>
                    <button
                        type="button"
                        onClick={() => setMode("demo")}
                        className={`px-3 py-1 rounded-full ${mode === "demo" ? "bg-[#0F172A] text-white" : "text-gray-500"}`}
                    >
                        Demo
                    </button>
                </div>
                <div className="text-xs text-gray-500">
                    Backend:{" "}
                    <span className={backendOk ? "text-green-600" : "text-red-500"}>
                        {backendOk ? "Connected" : "Offline"}
                    </span>
                </div>
                <div className="text-xs text-gray-400">
                    Last fetch: {formatTimestamp(lastFetched)}
                </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
                <div className="flex flex-col text-xs uppercase tracking-wider text-gray-400">
                    Portfolio
                    <input
                        className="mt-1 rounded-md border border-gray-200 px-2 py-1 text-sm text-gray-900"
                        value={portfolioId}
                        onChange={(event) => setPortfolioId(event.target.value)}
                    />
                </div>
                <div className="flex flex-col text-xs uppercase tracking-wider text-gray-400">
                    Benchmark
                    <input
                        className="mt-1 rounded-md border border-gray-200 px-2 py-1 text-sm text-gray-900"
                        value={benchmark}
                        onChange={(event) => setBenchmark(event.target.value.toUpperCase())}
                    />
                </div>
                <button
                    type="button"
                    onClick={refresh}
                    disabled={status === "loading"}
                    className="rounded-md border border-gray-200 px-3 py-2 text-sm font-semibold text-gray-700 hover:bg-gray-50 disabled:opacity-60"
                >
                    {status === "loading" ? "Refreshing..." : "Refresh data"}
                </button>
            </div>
        </div>
    );
}
