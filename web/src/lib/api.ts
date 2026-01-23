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

async function fetchJson<T>(path: string, fallback: T): Promise<T> {
    try {
        const res = await fetch(path, { cache: "no-store" });
        if (!res.ok) {
            console.warn(`Failed to fetch ${path}, using mock data`);
            return fallback;
        }
        return await res.json();
    } catch (error) {
        console.error("API Error:", error);
        return fallback;
    }
}

export async function getLatestRun(): Promise<RunManifest> {
    return fetchJson("/api/latest-run", MOCK_MANIFEST);
}

export async function getRunMetrics(): Promise<RunMetricsResponse> {
    return fetchJson("/api/metrics", MOCK_METRICS);
}

export async function getPortfolio(): Promise<PortfolioResponse> {
    return fetchJson("/api/portfolio", { portfolio: MOCK_PORTFOLIO, holdings: MOCK_HOLDINGS });
}

export async function getDefinitions(): Promise<DefinitionsRegistry> {
    return fetchJson("/api/definitions", MOCK_DEFINITIONS);
}

export async function getNexusState(): Promise<NexusState> {
    const [metrics, portfolio, definitions] = await Promise.all([
        getRunMetrics(),
        getPortfolio(),
        getDefinitions(),
    ]);

    return {
        manifest: metrics.manifest || MOCK_MANIFEST,
        summary: metrics.summary || MOCK_SUMMARY,
        equity_curve: metrics.equity_curve || MOCK_METRICS.equity_curve,
        performance: metrics.performance || MOCK_METRICS.performance,
        monthly_returns: metrics.monthly_returns || MOCK_METRICS.monthly_returns,
        holdings: portfolio.holdings || MOCK_HOLDINGS,
        risk: metrics.risk || MOCK_METRICS.risk,
        portfolio: portfolio.portfolio || MOCK_PORTFOLIO,
        definitions,
    };
}
