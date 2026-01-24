import type {
    DefinitionsRegistry,
    Holding,
    PortfolioMeta,
    PortfolioSummary,
    RunManifest,
    RunMetricsResponse,
} from "@/types/nexus";

export const MOCK_MANIFEST: RunManifest = {
    run_id: "mock-run-123",
    timestamp: new Date().toISOString(),
    input_hash: "abc",
    data_hash: "xyz",
    status: "completed",
    code_version: "dev",
};

export const MOCK_SUMMARY: PortfolioSummary = {
    source: "demo",
    twr: 0.125,
    mwr: 0.11,
    final_value: 125000,
    last_date: "2024-03-15",
    max_drawdown: -0.085,
    errors: [],
    run_id: "mock-run-123",
    input_hash: "abc",
    data_hash: "xyz",
    timestamp: new Date().toISOString(),
};

export const MOCK_METRICS: RunMetricsResponse = {
    manifest: MOCK_MANIFEST,
    summary: MOCK_SUMMARY,
    equity_curve: [
        { date: "2024-01-01", value: 100000 },
        { date: "2024-02-01", value: 105000 },
        { date: "2024-03-01", value: 110000 },
        { date: "2024-03-15", value: 125000 },
    ],
    performance: [
        { date: "2024-01-01", value: 100000, daily_return: 0, drawdown: 0 },
        { date: "2024-02-01", value: 105000, daily_return: 0.05, drawdown: 0 },
        { date: "2024-03-01", value: 110000, daily_return: 0.0476, drawdown: 0 },
        { date: "2024-03-15", value: 125000, daily_return: 0.1364, drawdown: 0 },
    ],
    monthly_returns: [
        { date: "2024-01-31", return: 0.02 },
        { date: "2024-02-29", return: 0.03 },
        { date: "2024-03-31", return: 0.05 },
    ],
    risk: {
        var_95: -0.018,
        cvar_95: -0.025,
        volatility: 0.15,
        sharpe: 1.8,
    },
};

export const MOCK_HOLDINGS: Holding[] = [
    { ticker: "SPY", weight: 0.4, value: 50000, shares: 120, price: 416.7 },
    { ticker: "QQQ", weight: 0.3, value: 37500, shares: 80, price: 468.75 },
    { ticker: "GLD", weight: 0.1, value: 12500, shares: 50, price: 250 },
    { ticker: "NVDA", weight: 0.2, value: 25000, shares: 25, price: 1000 },
];

export const MOCK_PORTFOLIO: PortfolioMeta = {
    id: 0,
    name: "Main Portfolio",
    currency: "USD",
    benchmark: "SPY",
};

export const MOCK_DEFINITIONS: DefinitionsRegistry = {};
