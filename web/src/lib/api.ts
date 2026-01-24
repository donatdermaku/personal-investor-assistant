import type {
    DefinitionsRegistry,
    NexusState,
    PortfolioResponse,
    RunManifest,
    RunMetricsResponse,
    RunListItem,
    RunCreateResponse,
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

export async function getRuns(): Promise<RunListItem[]> {
    if (!API_BASE) return [];
    const data = await fetchJson<{ runs: RunListItem[] }>(`${API_BASE}/runs`, true);
    return data?.runs ?? [];
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

export async function createRun(params: {
    runType: "demo" | "uploaded";
    portfolioId?: string;
    file?: File | null;
}): Promise<RunCreateResponse> {
    if (!API_BASE) {
        throw new Error("Backend URL is not configured.");
    }
    const formData = new FormData();
    formData.append("run_type", params.runType);
    formData.append("portfolio_id", params.portfolioId ?? "default");
    if (params.file) {
        formData.append("file", params.file);
    }
    const res = await fetch(`${API_BASE}/run`, {
        method: "POST",
        body: formData,
    });
    if (!res.ok) {
        let detail = "Run creation failed.";
        try {
            const payload = await res.json();
            if (payload?.detail) {
                if (typeof payload.detail === "string") {
                    detail = payload.detail;
                } else {
                    detail = payload.detail.message || detail;
                }
            }
        } catch {
            // ignore parse errors
        }
        throw new Error(detail);
    }
    return res.json();
}

export async function getNexusState(
    mode: NexusMode,
    portfolioId = "default",
    runId?: string | null
): Promise<{
    state: NexusState | null;
    empty: boolean;
    activeRunId?: string | null;
}> {
    if (mode === "demo") {
        return {
            state: {
                manifest: MOCK_MANIFEST,
                coverage_summary: null,
                summary: MOCK_SUMMARY,
                equity_curve: MOCK_METRICS.equity_curve,
                performance: MOCK_METRICS.performance,
                monthly_returns: MOCK_METRICS.monthly_returns,
                holdings: MOCK_HOLDINGS,
                risk: MOCK_METRICS.risk,
                attribution_summary: null,
                attribution_timeseries: [],
                risk_contribution: { summary: {}, contributions: [] },
                rolling_metrics: [],
                macro_regimes: [],
                macro: { status: "unavailable", missing_series: [], as_of: null, flags: [] },
                benchmark_comparison: null,
                benchmark_timeseries: [],
                risk_free_series: [],
                corporate_actions: [],
                portfolio: MOCK_PORTFOLIO,
                definitions: MOCK_DEFINITIONS,
            },
            empty: false,
            activeRunId: MOCK_MANIFEST.run_id,
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

    const resolvedRunId = runId || latest?.run_id || null;

    if (!resolvedRunId) {
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
            activeRunId: null,
        };
    }

    const metrics = await getRunMetrics(resolvedRunId);

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
            coverage_summary: metrics.coverage_summary ?? null,
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
            attribution_summary: metrics.attribution_summary ?? null,
            attribution_timeseries: metrics.attribution_timeseries || [],
            risk_contribution: metrics.risk_contribution ?? { summary: {}, contributions: [] },
            rolling_metrics: metrics.rolling_metrics || [],
            macro_regimes: metrics.macro_regimes || [],
            macro: metrics.macro ?? { status: "unavailable", missing_series: [], as_of: null, flags: [] },
            benchmark_comparison: metrics.benchmark_comparison ?? null,
            benchmark_timeseries: metrics.benchmark_timeseries || [],
            risk_free_series: metrics.risk_free_series || [],
            corporate_actions: metrics.corporate_actions || [],
            portfolio: portfolio?.portfolio ?? null,
            definitions,
        },
        empty: false,
        activeRunId: resolvedRunId,
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
