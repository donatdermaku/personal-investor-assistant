"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { NexusState } from "@/types/nexus";
import { createRun, getHealth, getNexusState, getRuns, NexusMode } from "@/lib/api";
import type { RunCreateResponse, RunListItem } from "@/types/nexus";

type NexusStatus = "idle" | "loading" | "ready" | "empty" | "error";

interface NexusContextValue {
    mode: NexusMode;
    setMode: (mode: NexusMode) => void;
    portfolioId: string;
    setPortfolioId: (value: string) => void;
    benchmark: string;
    setBenchmark: (value: string) => void;
    runId: string | null;
    setRunId: (value: string | null) => void;
    runs: RunListItem[];
    runCreatorOpen: boolean;
    openRunCreator: () => void;
    closeRunCreator: () => void;
    createRun: (params: { runType: "demo" | "uploaded"; file?: File | null }) => Promise<RunCreateResponse>;
    lastRunCreated: RunCreateResponse | null;
    clearRunCreated: () => void;
    status: NexusStatus;
    lastFetched: string | null;
    backendOk: boolean;
    error: string | null;
    loadingMessage: string | null;
    loadingProgress: number;
    state: NexusState | null;
    refresh: () => void;
    contextPanelOpen: boolean;
    toggleContextPanel: () => void;
}

const NexusContext = createContext<NexusContextValue | null>(null);

const STORAGE_KEYS = {
    mode: "nexus.mode",
    portfolioId: "nexus.portfolioId",
    benchmark: "nexus.benchmark",
    runId: "nexus.runId",
};

function readStorage(key: string, fallback: string) {
    if (typeof window === "undefined") return fallback;
    const stored = window.localStorage.getItem(key);
    return stored || fallback;
}

export function NexusProvider({ children }: { children: React.ReactNode }) {
    const [mode, setMode] = useState<NexusMode>("live");
    const [portfolioId, setPortfolioId] = useState("default");
    const [benchmark, setBenchmark] = useState("SPY");
    const [runId, setRunId] = useState<string | null>(null);
    const [runs, setRuns] = useState<RunListItem[]>([]);
    const [runCreatorOpen, setRunCreatorOpen] = useState(false);
    const [lastRunCreated, setLastRunCreated] = useState<RunCreateResponse | null>(null);
    const [status, setStatus] = useState<NexusStatus>("idle");
    const [state, setState] = useState<NexusState | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loadingMessage, setLoadingMessage] = useState<string | null>(null);
    const [loadingProgress, setLoadingProgress] = useState(0);
    const [lastFetched, setLastFetched] = useState<string | null>(null);
    const [backendOk, setBackendOk] = useState(false);
    const [refreshKey, setRefreshKey] = useState(0);
    const [contextPanelOpen, setContextPanelOpen] = useState(false);

    /* eslint-disable react-hooks/set-state-in-effect */
    useEffect(() => {
        // Hydrate from localStorage after mount to avoid SSR/client mismatch.
        const storedMode = readStorage(STORAGE_KEYS.mode, "live") as NexusMode;
        const storedPortfolio = readStorage(STORAGE_KEYS.portfolioId, "default");
        const storedBenchmark = readStorage(STORAGE_KEYS.benchmark, "SPY");
        const storedRunId = readStorage(STORAGE_KEYS.runId, "");
        setMode(storedMode);
        setPortfolioId(storedPortfolio);
        setBenchmark(storedBenchmark);
        setRunId(storedRunId ? storedRunId : null);
    }, []);
    /* eslint-enable react-hooks/set-state-in-effect */

    useEffect(() => {
        if (typeof window === "undefined") return;
        window.localStorage.setItem(STORAGE_KEYS.mode, mode);
    }, [mode]);

    useEffect(() => {
        if (typeof window === "undefined") return;
        window.localStorage.setItem(STORAGE_KEYS.portfolioId, portfolioId);
    }, [portfolioId]);

    useEffect(() => {
        if (typeof window === "undefined") return;
        window.localStorage.setItem(STORAGE_KEYS.benchmark, benchmark);
    }, [benchmark]);

    useEffect(() => {
        if (typeof window === "undefined") return;
        if (!runId) {
            window.localStorage.removeItem(STORAGE_KEYS.runId);
            return;
        }
        window.localStorage.setItem(STORAGE_KEYS.runId, runId);
    }, [runId]);

    useEffect(() => {
        let active = true;
        const load = async () => {
            setStatus("loading");
            setError(null);
            setLoadingMessage("Checking backend health...");
            setLoadingProgress(15);
            const health = await getHealth();
            if (!active) return;
            setBackendOk(health);

            try {
                setLoadingMessage(mode === "live" ? "Loading available runs..." : "Preparing demo context...");
                setLoadingProgress(35);
                const runsList = mode === "live" ? await getRuns() : [];
                if (!active) return;
                setRuns(runsList);
                const runIdValid = runId ? runsList.some((run) => run.run_id === runId) : false;
                const resolvedRunId = runIdValid ? runId : runsList[0]?.run_id || null;
                setLoadingMessage("Loading portfolio analytics...");
                setLoadingProgress(70);
                const response = await getNexusState(mode, portfolioId, resolvedRunId);
                if (!active) return;
                setState(response.state);
                setStatus(response.empty ? "empty" : "ready");
                setLoadingMessage(null);
                setLoadingProgress(100);
                setLastFetched(new Date().toISOString());
                if (!runIdValid && response.activeRunId) {
                    setRunId(response.activeRunId);
                }
            } catch (err) {
                if (!active) return;
                setStatus("error");
                const message = err instanceof Error ? err.message : "Failed to load data.";
                setError(health ? message : "Backend is offline. Start the API service or switch to demo mode.");
                setLoadingMessage(null);
            }
        };
        load();
        return () => {
            active = false;
        };
    }, [mode, portfolioId, runId, refreshKey]);

    const value = useMemo(
        () => ({
            mode,
            setMode,
            portfolioId,
            setPortfolioId,
            benchmark,
            setBenchmark,
            runId,
            setRunId,
            runs,
            runCreatorOpen,
            openRunCreator: () => setRunCreatorOpen(true),
            closeRunCreator: () => setRunCreatorOpen(false),
            createRun: async ({ runType, file }: { runType: "demo" | "uploaded"; file?: File | null }) => {
                const result = await createRun({ runType, file, portfolioId });
                setLastRunCreated(result);
                setRunId(result.run_id);
                setRunCreatorOpen(false);
                setRefreshKey((prev) => prev + 1);
                return result;
            },
            lastRunCreated,
            clearRunCreated: () => setLastRunCreated(null),
            status,
            lastFetched,
            backendOk,
            error,
            loadingMessage,
            loadingProgress,
            state,
            refresh: () => setRefreshKey((prev) => prev + 1),
            contextPanelOpen,
            toggleContextPanel: () => setContextPanelOpen((prev) => !prev),
        }),
        [
            mode,
            portfolioId,
            benchmark,
            runId,
            runs,
            runCreatorOpen,
            lastRunCreated,
            status,
            lastFetched,
            backendOk,
            error,
            loadingMessage,
            loadingProgress,
            state,
            contextPanelOpen,
        ]
    );

    return <NexusContext.Provider value={value}>{children}</NexusContext.Provider>;
}

export function useNexus() {
    const context = useContext(NexusContext);
    if (!context) {
        throw new Error("useNexus must be used within NexusProvider");
    }
    return context;
}
