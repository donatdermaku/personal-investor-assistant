export interface RunManifest {
    run_id: string;
    timestamp: string | null;
    input_hash: string | null;
    data_hash: string | null;
    status?: "running" | "completed" | "failed";
    code_version?: string | null;
    coverage_summary?: CoverageSummaryDetailed | Record<string, CoverageSummary>;
    meta?: Record<string, unknown>;
}

export interface RunListItem {
    run_id: string;
    status: string;
    timestamp: string | null;
    input_hash?: string | null;
    data_hash?: string | null;
}

export interface RunCreateResponse {
    run_id: string;
    status: string;
    timestamp: string | null;
}

export interface PortfolioSummary {
    source: string;
    twr: number | null;
    mwr: number | null; // Personal return (IRR)
    final_value: number | null;
    last_date: string | null;
    max_drawdown: number | null;
    errors: string[];
    run_id?: string;
    input_hash?: string;
    data_hash?: string;
    timestamp?: string;
}

export interface TimeSeriesPoint {
    date: string;
    value: number;
}

export interface PerformancePoint {
    date: string;
    value: number | null;
    cash?: number | null;
    daily_return?: number | null;
    drawdown?: number | null;
}

export interface MonthlyReturnPoint {
    date: string;
    return: number | null;
}

export interface Holding {
    ticker: string;
    shares?: number;
    price?: number;
    value?: number;
    weight?: number; // 0-1
    sector?: string;
    cost_basis?: number;
}

export interface RiskMetrics {
    var_95: number | null;
    cvar_95: number | null;
    volatility: number | null;
    sharpe: number | null;
}

export interface AttributionSummary {
    total_return: number;
    allocation: number;
    selection: number;
    interaction: number;
    method?: string;
    benchmark?: string;
    per_asset?: Array<{
        ticker: string;
        allocation: number;
        selection: number;
        interaction: number;
        total: number;
    }>;
}

export interface AttributionTimeseriesPoint {
    date: string;
    allocation: number;
    selection: number;
    interaction: number;
    total_return: number;
}

export interface RiskContributionSummary {
    portfolio_volatility?: number | null;
    portfolio_var?: number | null;
    var_alpha?: number | null;
}

export interface RiskContributionItem {
    ticker: string;
    volatility_contribution: number;
    volatility_pct: number;
    var_contribution: number;
    var_pct: number;
}

export interface RiskContributionPayload {
    summary: RiskContributionSummary;
    contributions: RiskContributionItem[];
}

export interface RollingMetricPoint {
    date: string;
    rolling_volatility: number | null;
    rolling_sharpe: number | null;
    rolling_drawdown: number | null;
}

export interface MacroRegimeFlag {
    date: string;
    inflation_yoy?: number | null;
    fed_funds?: number | null;
    vix?: number | null;
    rates_change_6m?: number | null;
    high_inflation?: boolean | number;
    rising_rates?: boolean | number;
    risk_off?: boolean | number;
}

export interface MacroPayload {
    status: "ok" | "partial" | "unavailable";
    missing_series: string[];
    as_of: string | null;
    flags: MacroRegimeFlag[];
}

export interface BenchmarkComparisonSummary {
    tracking_error?: number | null;
    portfolio_volatility?: number | null;
    benchmark_volatility?: number | null;
    correlation?: number | null;
    tracking_error_implied?: number | null;
}

export interface BenchmarkTimeseriesPoint {
    date: string;
    portfolio_return: number | null;
    benchmark_return: number | null;
    active_return: number | null;
    relative_drawdown: number | null;
}

export interface CoverageSummary {
    total: number;
    covered: number;
    last_date?: string | null;
    missing_count?: number;
}

export interface CoveragePolicy {
    min_score_for_kpis: number;
    min_history_days: number;
    max_gap_days: number;
}

export interface CoveragePerTicker {
    score: number;
    history_days: number;
    missing_days: number;
    largest_gap_days: number;
    status: string;
    reason_codes: string[];
}

export interface CoverageAggregate {
    coverage_ratio: number;
    min_ticker_score: number;
    benchmark_score: number | null;
    rf_score: number | null;
}

export interface CoverageSummaryDetailed {
    as_of: string | null;
    status: "sufficient" | "insufficient" | "unknown";
    score: number;
    policy: CoveragePolicy;
    required: {
        tickers: string[];
        history_days_needed: number;
    };
    per_ticker: Record<string, CoveragePerTicker>;
    aggregate: CoverageAggregate;
    reason_codes: string[];
    contract_version?: string;
}

export interface PortfolioMeta {
    id: number;
    name: string;
    currency: string;
    benchmark?: string | null;
}

export interface DefinitionItem {
    title: string;
    definition_md: string;
    assumptions?: string;
    warnings?: string;
}

export type DefinitionsRegistry = Record<string, DefinitionItem>;

export interface NexusState {
    manifest: RunManifest;
    coverage_summary?: CoverageSummaryDetailed | null;
    summary: PortfolioSummary;
    equity_curve: TimeSeriesPoint[];
    performance: PerformancePoint[];
    monthly_returns: MonthlyReturnPoint[];
    holdings: Holding[];
    risk: RiskMetrics;
    attribution_summary?: AttributionSummary | null;
    attribution_timeseries?: AttributionTimeseriesPoint[];
    risk_contribution?: RiskContributionPayload | null;
    rolling_metrics?: RollingMetricPoint[];
    macro_regimes?: MacroRegimeFlag[];
    macro?: MacroPayload | null;
    benchmark_comparison?: BenchmarkComparisonSummary | null;
    benchmark_timeseries?: BenchmarkTimeseriesPoint[];
    portfolio: PortfolioMeta | null;
    definitions?: DefinitionsRegistry;
}

export interface RunMetricsResponse {
    manifest: RunManifest;
    coverage_summary?: CoverageSummaryDetailed | null;
    summary: PortfolioSummary;
    equity_curve: TimeSeriesPoint[];
    performance: PerformancePoint[];
    monthly_returns: MonthlyReturnPoint[];
    risk: RiskMetrics;
    attribution_summary?: AttributionSummary | null;
    attribution_timeseries?: AttributionTimeseriesPoint[];
    risk_contribution?: RiskContributionPayload | null;
    rolling_metrics?: RollingMetricPoint[];
    macro_regimes?: MacroRegimeFlag[];
    macro?: MacroPayload | null;
    benchmark_comparison?: BenchmarkComparisonSummary | null;
    benchmark_timeseries?: BenchmarkTimeseriesPoint[];
}

export interface PortfolioResponse {
    portfolio: PortfolioMeta;
    holdings: Holding[];
}
