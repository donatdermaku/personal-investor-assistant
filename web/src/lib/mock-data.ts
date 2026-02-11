import type {
    DefinitionsRegistry,
    Holding,
    PortfolioMeta,
    PortfolioSummary,
    RunManifest,
    RunMetricsResponse,
    DiagnosticSignal,
    RiskMetrics,
    RiskContributionPayload,
    RollingMetricPoint,
    AttributionTimeseriesPoint,
    MacroRegimeFlag,
    AttributionSummary,
} from "@/types/nexus";

// --- HELPERS ---
function generateRandomWalk(days: number, startValue: number, volatility: number, drift: number) {
    const data = [];
    let currentValue = startValue;
    const dailyDrift = drift / 252;
    const dailyVol = volatility / Math.sqrt(252);

    // Start date: 1 year ago
    let currentDate = new Date();
    currentDate.setFullYear(currentDate.getFullYear() - 1);

    for (let i = 0; i < days; i++) {
        // Skip weekends
        while (currentDate.getDay() === 0 || currentDate.getDay() === 6) {
            currentDate.setDate(currentDate.getDate() + 1);
        }

        const dateStr = currentDate.toISOString().split('T')[0];

        // Geometric Brownian Motion step
        const change = (dailyDrift + dailyVol * (Math.random() - 0.5) * 2); // Simplified normal dist
        const newValue = currentValue * (1 + change);

        data.push({
            date: dateStr,
            value: Number(newValue.toFixed(2)),
            daily_return: Number(change),
            drawdown: 0 // calculated later
        });

        currentValue = newValue;
        currentDate.setDate(currentDate.getDate() + 1);
    }

    // Calculate drawdowns
    let peak = startValue;
    return data.map(point => {
        if (point.value > peak) peak = point.value;
        const drawdown = (point.value - peak) / peak;
        return { ...point, drawdown: Number(drawdown) };
    });
}

// --- DATA GENERATION ---

// 1. Performance History (1 Year)
const EQUITY_SERIES = generateRandomWalk(252, 100000, 0.18, 0.12);
const FINAL_VALUE = EQUITY_SERIES[EQUITY_SERIES.length - 1].value;
const TOTAL_RETURN = (FINAL_VALUE - 100000) / 100000;
const MAX_DRAWDOWN = Math.min(...EQUITY_SERIES.map(p => p.drawdown));

// 2. Rolling Metrics (Derived from Equity Series)
const ROLLING_METRICS: RollingMetricPoint[] = EQUITY_SERIES.map((point, i) => {
    // Simulate rolling volatility and sharpe oscillating
    const noise = Math.sin(i / 20) * 0.05;
    return {
        date: point.date,
        rolling_volatility: 0.15 + noise + (Math.random() * 0.02),
        rolling_sharpe: 1.5 + (noise * 5) + (Math.random() * 0.5),
        rolling_drawdown: point.drawdown
    };
});

// 3. Attribution Timeseries (Simulated)
const ATTRIBUTION_TIMESERIES: AttributionTimeseriesPoint[] = EQUITY_SERIES.map((point, i) => {
    // Cumulative attribution effects
    const progress = i / 252;
    return {
        date: point.date,
        allocation: 0.05 * progress + (Math.sin(i / 10) * 0.01),
        selection: 0.08 * progress + (Math.cos(i / 15) * 0.01),
        interaction: -0.01 * progress + (Math.random() * 0.005),
        total_return: (point.value - 100000) / 100000
    };
});

// 4. Macro Regimes (Simulated Trends)
const MACRO_REGIMES: MacroRegimeFlag[] = EQUITY_SERIES.map((point, i) => {
    // Slow moving macro trends
    return {
        date: point.date,
        inflation_yoy: 0.03 + (Math.cos(i / 100) * 0.01), // 3-4% Inflation
        fed_funds: 5.25 + (Math.sin(i / 200) * 0.25), // Rates 5.00-5.50%
        vix: 15 + (Math.random() * 5) + (point.drawdown < -0.05 ? 10 : 0), // Spike VIX on drawdowns
        risk_off: point.drawdown < -0.05
    };
});

// 5. Holdings (Diverse Mix)
export const MOCK_HOLDINGS: Holding[] = [
    { ticker: "SPY", weight: 0.35, value: FINAL_VALUE * 0.35, shares: 75, price: 470.50 },
    { ticker: "QQQ", weight: 0.25, value: FINAL_VALUE * 0.25, shares: 60, price: 405.20 },
    { ticker: "NVDA", weight: 0.12, value: FINAL_VALUE * 0.12, shares: 25, price: 850.00 },
    { ticker: "MSFT", weight: 0.08, value: FINAL_VALUE * 0.08, shares: 20, price: 410.00 },
    { ticker: "GLD", weight: 0.05, value: FINAL_VALUE * 0.05, shares: 30, price: 205.00 },
    { ticker: "BTC-USD", weight: 0.05, value: FINAL_VALUE * 0.05, shares: 0.15, price: 65000.00 },
    { ticker: "TLT", weight: 0.05, value: FINAL_VALUE * 0.05, shares: 55, price: 92.00 },
    { ticker: "JPM", weight: 0.03, value: FINAL_VALUE * 0.03, shares: 15, price: 185.00 },
    { ticker: "CASH", weight: 0.02, value: FINAL_VALUE * 0.02, shares: 1, price: FINAL_VALUE * 0.02 },
];

export const MOCK_MANIFEST: RunManifest = {
    run_id: "demo-run-adv",
    timestamp: new Date().toISOString(),
    input_hash: "demo-dataset-adv",
    data_hash: "market-data-adv",
    status: "completed",
    code_version: "production",
    new_run_created: true,
};

export const MOCK_SUMMARY: PortfolioSummary = {
    source: "demo",
    twr: Number(TOTAL_RETURN),
    mwr: Number(TOTAL_RETURN - 0.01), // Slight difference for realism
    final_value: FINAL_VALUE,
    last_date: new Date().toISOString().split('T')[0],
    max_drawdown: Number(MAX_DRAWDOWN),
    errors: [],
    run_id: "demo-run-adv",
    input_hash: "demo",
    data_hash: "demo",
    timestamp: new Date().toISOString(),
};

const MOCK_RISK: RiskMetrics = {
    var_95: -0.021,
    cvar_95: -0.035,
    volatility: 0.145,
    sharpe: 1.92,
    beta: 1.15,
};

const MOCK_DIAGNOSTICS: DiagnosticSignal[] = [
    {
        key: "cash_drag",
        category: "risk",
        severity: "low",
        summary: "Low cash drag detected.",
        message: "Portfolio holds 2% cash. Consider deploying for full exposure.",
        evidence: ["Cash weight: 2.0%"],
        metrics_used: ["holdings"],
        as_of: new Date().toISOString(),
        confidence: 0.95
    },
    {
        key: "concentration_tech",
        category: "risk",
        severity: "medium",
        summary: "Tech sector concentration detected.",
        message: "High exposure to Technology (QQQ, NVDA, MSFT) exceeds 40%.",
        evidence: ["Tech weight: 45.0%"],
        metrics_used: ["holdings.sector"],
        as_of: new Date().toISOString(),
        confidence: 0.90
    },
];

const MOCK_ATTRIBUTION: AttributionSummary = {
    total_return: TOTAL_RETURN,
    allocation: 0.045,
    selection: 0.082,
    interaction: -0.005,
    per_asset: [
        { ticker: "Allocation", allocation: 0.045, selection: 0, interaction: 0, total: 0.045 },
        { ticker: "Selection", allocation: 0, selection: 0.082, interaction: 0, total: 0.082 },
    ]
};

const MOCK_RISK_CONTRIBUTION: RiskContributionPayload = {
    summary: { // Removed the dictionary, using proper summary stats if needed, otherwise empty
        portfolio_volatility: 0.145,
        portfolio_var: -0.021,
        var_alpha: 0.05
    },
    contributions: [
        { ticker: "Technology", volatility_contribution: 0.08, volatility_pct: 0.55, var_contribution: -0.012, var_pct: 0.55 },
        { ticker: "Finance", volatility_contribution: 0.015, volatility_pct: 0.10, var_contribution: -0.002, var_pct: 0.10 },
        { ticker: "Commodities", volatility_contribution: 0.007, volatility_pct: 0.05, var_contribution: -0.001, var_pct: 0.05 },
        { ticker: "Crypto", volatility_contribution: 0.022, volatility_pct: 0.15, var_contribution: -0.003, var_pct: 0.15 },
        { ticker: "Bonds", volatility_contribution: 0.007, volatility_pct: 0.05, var_contribution: -0.001, var_pct: 0.05 },
    ]
};

const ASSETS = ["SPY", "QQQ", "NVDA", "GLD", "BTC"];
const CORRELATION_MATRIX_DATA = [
    [1.0, 0.85, 0.65, 0.10, 0.45],
    [0.85, 1.0, 0.75, 0.05, 0.55],
    [0.65, 0.75, 1.0, 0.15, 0.60],
    [0.10, 0.05, 0.15, 1.0, 0.20],
    [0.45, 0.55, 0.60, 0.20, 1.0],
];

// Convert matrix to Record<string, Record<string, number>>
const CORRELATION_RECORD: Record<string, Record<string, number>> = {};
ASSETS.forEach((rowTicker, i) => {
    CORRELATION_RECORD[rowTicker] = {};
    ASSETS.forEach((colTicker, j) => {
        CORRELATION_RECORD[rowTicker][colTicker] = CORRELATION_MATRIX_DATA[i][j];
    });
});

export const MOCK_METRICS: RunMetricsResponse = {
    manifest: MOCK_MANIFEST,
    summary: MOCK_SUMMARY,
    equity_curve: EQUITY_SERIES as any,
    performance: EQUITY_SERIES as any,
    monthly_returns: [
        { date: "2023-04-30", return: 0.02 },
        { date: "2023-05-31", return: 0.015 },
        { date: "2023-06-30", return: 0.03 },
        { date: "2023-07-31", return: 0.04 },
        { date: "2023-08-31", return: -0.02 },
        { date: "2023-09-30", return: -0.04 },
        { date: "2023-10-31", return: 0.01 },
        { date: "2023-11-30", return: 0.06 },
        { date: "2023-12-31", return: 0.03 },
        { date: "2024-01-31", return: 0.02 },
        { date: "2024-02-29", return: 0.05 },
        { date: "2024-03-31", return: 0.035 },
    ],
    risk: MOCK_RISK,
    diagnostics: MOCK_DIAGNOSTICS,
    attribution_summary: MOCK_ATTRIBUTION,
    attribution_timeseries: ATTRIBUTION_TIMESERIES,
    risk_contribution: MOCK_RISK_CONTRIBUTION,
    correlation_matrix: {
        status: "sufficient",
        n_obs: 252,
        assets_included: ASSETS,
        assets_excluded: [],
        matrix: CORRELATION_RECORD
    },
    rolling_metrics: ROLLING_METRICS,
    macro_regimes: MACRO_REGIMES,
    benchmark_comparison: {
        tracking_error: 0.065,
        portfolio_volatility: 0.145,
        benchmark_volatility: 0.12,
        correlation: 0.85
    },
};

export const MOCK_PORTFOLIO: PortfolioMeta = {
    id: 0,
    name: "Growth Demo Portfolio",
    currency: "USD",
    benchmark: "SPY",
};

export const MOCK_DEFINITIONS: DefinitionsRegistry = {
    "twr": {
        title: "Time-Weighted Return",
        definition_md: "The compound rate of growth of $1 over the period, eliminating the effects of cash flow timing."
    },
    "mwr": {
        title: "Money-Weighted Return",
        definition_md: "Internal Rate of Return (IRR) that accounts for the timing and size of cash flows."
    },
    "max_drawdown": {
        title: "Maximum Drawdown",
        definition_md: "The largest peak-to-trough decline in portfolio value during the period."
    },
    "sharpe": {
        title: "Sharpe Ratio",
        definition_md: "Measure of risk-adjusted return, calculated as excess return per unit of volatility."
    },
    "var_daily": {
        title: "Value at Risk (95% Daily)",
        definition_md: "The minimum expected loss for the worst 5% of days."
    },
    "cvar_daily": {
        title: "Conditional VaR",
        definition_md: "The average loss on days exceeding the VaR threshold (Tail Risk)."
    },
    "rolling_volatility": {
        title: "Rolling Volatility",
        definition_md: "Standard deviation of returns calculated over a sliding window (e.g., 30 days)."
    },
};
