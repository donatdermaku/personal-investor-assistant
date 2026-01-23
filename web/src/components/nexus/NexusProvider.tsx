"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { NexusState } from "@/types/nexus";
import { getHealth, getNexusState, NexusMode } from "@/lib/api";

type NexusStatus = "idle" | "loading" | "ready" | "empty" | "error";

interface NexusContextValue {
    mode: NexusMode;
    setMode: (mode: NexusMode) => void;
    portfolioId: string;
    setPortfolioId: (value: string) => void;
    benchmark: string;
    setBenchmark: (value: string) => void;
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
        let active = true;
        const load = async () => {
            setStatus("loading");
            setError(null);
            const health = await getHealth();
            if (!active) return;
            setBackendOk(health);

            try {
                const response = await getNexusState(mode, portfolioId);
                if (!active) return;
                setState(response.state);
                setStatus(response.empty ? "empty" : "ready");
                setLastFetched(new Date().toISOString());
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
    }, [mode, portfolioId, refreshKey]);

    const value = useMemo(
        () => ({
            mode,
            setMode,
            portfolioId,
            setPortfolioId,
            benchmark,
            setBenchmark,
            status,
            lastFetched,
            backendOk,
            error,
            state,
            refresh: () => setRefreshKey((prev) => prev + 1),
        }),
        [mode, portfolioId, benchmark, status, lastFetched, backendOk, error, state]
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
