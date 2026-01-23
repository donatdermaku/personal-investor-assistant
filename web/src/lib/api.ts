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

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function fetchJson<T>(url: string, fallback: T): Promise<T> {
    try {
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) {
            if (res.status === 404) {
                return fallback;
            }
            console.warn(`Failed to fetch ${url}, using fallback`);
            return fallback;
        }
        return await res.json();
    } catch (error) {
        console.error("API Error:", error);
        return fallback;
    }
}

export async function getLatestRun(): Promise<RunManifest> {
    return fetchJson(`${API_BASE}/latest-run`, MOCK_MANIFEST);
}

export async function getDefinitions(): Promise<DefinitionsRegistry> {
    return fetchJson(`${API_BASE}/definitions`, MOCK_DEFINITIONS);
}

export async function getPortfolio(portfolioId = "default"): Promise<PortfolioResponse> {
    return fetchJson(
        `${API_BASE}/portfolio/${portfolioId}`,
        { portfolio: MOCK_PORTFOLIO, holdings: MOCK_HOLDINGS }
    );
}

export async function getRunMetrics(runId: string): Promise<RunMetricsResponse> {
    return fetchJson(`${API_BASE}/run/${runId}`, MOCK_METRICS);
}

export async function getNexusState(): Promise<NexusState> {
    const latest = await getLatestRun();
    const [definitions, portfolio] = await Promise.all([
        getDefinitions(),
        getPortfolio(),
    ]);

    if (!latest || latest.run_id === MOCK_MANIFEST.run_id) {
        return {
            manifest: MOCK_MANIFEST,
            summary: MOCK_SUMMARY,
            equity_curve: MOCK_METRICS.equity_curve,
            performance: MOCK_METRICS.performance,
            monthly_returns: MOCK_METRICS.monthly_returns,
            holdings: portfolio.holdings || MOCK_HOLDINGS,
            risk: MOCK_METRICS.risk,
            portfolio: portfolio.portfolio || MOCK_PORTFOLIO,
            definitions,
        };
    }

    const metrics = await getRunMetrics(latest.run_id);

    return {
        manifest: metrics.manifest || latest,
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
