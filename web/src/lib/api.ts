import type {
    DefinitionsRegistry,
    NexusState,
    PortfolioResponse,
    RunManifest,
    RunMetricsResponse,
} from "@/types/nexus";
import {
    MOCK_DEFINITIONS,
    MOCK_HOLDINGS,
    MOCK_MANIFEST,
    MOCK_METRICS,
    MOCK_PORTFOLIO,
    MOCK_SUMMARY,
} from "@/lib/mock-data";

export type NexusMode = "live" | "demo";

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

async function fetchJson<T>(url: string, allow404 = false): Promise<T | null> {
    const res = await fetch(url, { cache: "no-store" });
    if (res.status === 404 && allow404) {
        return null;
    }
    if (!res.ok) {
        throw new Error(`Request failed (${res.status})`);
    }
    return res.json();
}

export async function getHealth(): Promise<boolean> {
    if (!API_BASE) return false;
    try {
        const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
        return res.ok;
    } catch {
        return false;
    }
}

export async function getLatestRun(): Promise<RunManifest | null> {
    if (!API_BASE) return null;
    return fetchJson(`${API_BASE}/latest-run`, true);
}

export async function getDefinitions(): Promise<DefinitionsRegistry> {
    if (!API_BASE) return {};
    const data = await fetchJson<DefinitionsRegistry>(`${API_BASE}/definitions`);
    return data ?? {};
}

export async function getPortfolio(portfolioId = "default"): Promise<PortfolioResponse | null> {
    if (!API_BASE) return null;
    return fetchJson(`${API_BASE}/portfolio/${portfolioId}`, true);
}

export async function getRunMetrics(runId: string): Promise<RunMetricsResponse> {
    if (!API_BASE) {
        throw new Error("Backend URL is not configured.");
    }
    const data = await fetchJson<RunMetricsResponse>(`${API_BASE}/run/${runId}`);
    if (!data) {
        throw new Error("Run metrics unavailable.");
    }
    return data;
}

export async function getNexusState(
    mode: NexusMode,
    portfolioId = "default"
): Promise<{
    state: NexusState | null;
    empty: boolean;
}> {
    if (mode === "demo") {
        return {
            state: {
                manifest: MOCK_MANIFEST,
                summary: MOCK_SUMMARY,
                equity_curve: MOCK_METRICS.equity_curve,
                performance: MOCK_METRICS.performance,
                monthly_returns: MOCK_METRICS.monthly_returns,
                holdings: MOCK_HOLDINGS,
                risk: MOCK_METRICS.risk,
                portfolio: MOCK_PORTFOLIO,
                definitions: MOCK_DEFINITIONS,
            },
            empty: false,
        };
    }

    if (!API_BASE) {
        throw new Error("Backend URL is not configured.");
    }

    const [latest, definitions, portfolio] = await Promise.all([
        getLatestRun(),
        getDefinitions(),
        getPortfolio(portfolioId),
    ]);

    if (!latest) {
        return {
            state: {
                manifest: {
                    run_id: "",
                    timestamp: null,
                    input_hash: null,
                    data_hash: null,
                },
                summary: {
                    source: "",
                    twr: null,
                    mwr: null,
                    final_value: null,
                    last_date: null,
                    max_drawdown: null,
                    errors: [],
                },
                equity_curve: [],
                performance: [],
                monthly_returns: [],
                holdings: portfolio?.holdings ?? [],
                risk: {
                    var_95: null,
                    cvar_95: null,
                    volatility: null,
                    sharpe: null,
                },
                portfolio: portfolio?.portfolio ?? null,
                definitions,
            },
            empty: true,
        };
    }

    const metrics = await getRunMetrics(latest.run_id);

    const summary = metrics.summary || {
        source: "",
        twr: null,
        mwr: null,
        final_value: null,
        last_date: null,
        max_drawdown: null,
        errors: [],
    };

    return {
        state: {
            manifest: metrics.manifest || latest,
            summary,
            equity_curve: metrics.equity_curve || [],
            performance: metrics.performance || [],
            monthly_returns: metrics.monthly_returns || [],
            holdings: portfolio?.holdings ?? [],
            risk: metrics.risk || {
                var_95: null,
                cvar_95: null,
                volatility: null,
                sharpe: null,
            },
            portfolio: portfolio?.portfolio ?? null,
            definitions,
        },
        empty: false,
    };
}

export async function downloadExport(runId: string, artifact: string): Promise<void> {
    if (!API_BASE) {
        throw new Error("Backend URL is not configured.");
    }
    const res = await fetch(`${API_BASE}/run/${runId}/export/${artifact}`);
    if (!res.ok) {
        if (res.status === 404) {
            throw new Error("Artifact not available.");
        }
        throw new Error("Export failed.");
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${artifact}-${runId}`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}
