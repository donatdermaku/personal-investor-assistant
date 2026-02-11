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

type ApiErrorPayload = {
    detail?: string | {
        message?: string;
        hint?: string;
        details?: {
            errors?: string[];
        };
    };
};

function fallbackErrorMessage(status: number): string {
    if (status === 400) return "The request is invalid. Check your inputs and retry.";
    if (status === 401 || status === 403) return "Access denied for this action.";
    if (status === 404) return "Requested data was not found.";
    if (status === 422) return "Input validation failed. Review data and retry.";
    if (status >= 500) return "The backend failed while processing the request. Retry shortly.";
    return "Request failed.";
}

async function resolveErrorMessage(res: Response): Promise<string> {
    const fallback = fallbackErrorMessage(res.status);
    try {
        const payload = (await res.json()) as ApiErrorPayload;
        if (typeof payload?.detail === "string" && payload.detail.trim().length > 0) {
            return payload.detail;
        }
        if (payload?.detail && typeof payload.detail === "object") {
            const message = payload.detail.message?.trim();
            const hint = payload.detail.hint?.trim();
            const validationErrors = payload.detail.details?.errors ?? [];
            if (message && hint) {
                return `${message} ${hint}`;
            }
            if (message) {
                return message;
            }
            if (validationErrors.length > 0) {
                return `CSV validation failed: ${validationErrors.slice(0, 3).join("; ")}`;
            }
        }
    } catch {
        // Fallback to status-based message.
    }
    return fallback;
}

async function fetchJson<T>(url: string, allow404 = false): Promise<T | null> {
    const res = await fetch(url, { cache: "no-store" });
    if (res.status === 404 && allow404) {
        return null;
    }
    if (!res.ok) {
        const message = await resolveErrorMessage(res);
        throw new Error(message);
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
        const message = await resolveErrorMessage(res);
        throw new Error(message);
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
                diagnostics: [],
                correlation_matrix: null,
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
                data_contracts: {},
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
            diagnostics: metrics.diagnostics ?? [],
            correlation_matrix: metrics.correlation_matrix ?? null,
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
            data_contracts: metrics.data_contracts ?? null,
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
        const message = await resolveErrorMessage(res);
        throw new Error(message);
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
