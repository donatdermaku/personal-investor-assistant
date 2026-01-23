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
    state: NexusState | null;
    refresh: () => void;
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
    const [lastFetched, setLastFetched] = useState<string | null>(null);
    const [backendOk, setBackendOk] = useState(false);
    const [refreshKey, setRefreshKey] = useState(0);

    useEffect(() => {
        setMode(readStorage(STORAGE_KEYS.mode, "live") as NexusMode);
        setPortfolioId(readStorage(STORAGE_KEYS.portfolioId, "default"));
        setBenchmark(readStorage(STORAGE_KEYS.benchmark, "SPY"));
        setRunId(readStorage(STORAGE_KEYS.runId, ""));
    }, []);

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
            const health = await getHealth();
            if (!active) return;
            setBackendOk(health);

            try {
                const runsList = mode === "live" ? await getRuns() : [];
                if (!active) return;
                setRuns(runsList);
                const runIdValid = runId ? runsList.some((run) => run.run_id === runId) : false;
                const resolvedRunId = runIdValid ? runId : runsList[0]?.run_id || null;
                const response = await getNexusState(mode, portfolioId, resolvedRunId);
                if (!active) return;
                setState(response.state);
                setStatus(response.empty ? "empty" : "ready");
                setLastFetched(new Date().toISOString());
                if (!runIdValid && response.activeRunId) {
                    setRunId(response.activeRunId);
                }
            } catch (err) {
                if (!active) return;
                setStatus("error");
                setError(err instanceof Error ? err.message : "Failed to load data.");
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
            state,
            refresh: () => setRefreshKey((prev) => prev + 1),
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
            state,
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
