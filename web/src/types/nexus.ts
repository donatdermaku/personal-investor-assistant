export interface RunManifest {
    run_id: string;
    timestamp: string | null;
    input_hash: string | null;
    data_hash: string | null;
    status?: "running" | "completed" | "failed";
    code_version?: string | null;
    coverage_summary?: CoverageSummaryDetailed | Record<string, CoverageSummary>;
    meta?: Record<string, unknown>;
    new_run_created?: boolean;
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
    warnings?: {
        failed_tickers?: {
            count: number;
            tickers: string[];
            message: string;
        };
    };
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
    beta: number | null;
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
    status: "sufficient" | "partial" | "unavailable";
    missing_series: string[];
    as_of: string | null;
    flags: MacroRegimeFlag[];
    available_series?: string[];
    tags?: string[];
    warnings?: string[];
    cache_status?: Record<string, string>;
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

export interface RiskFreeSeriesPoint {
    date: string;
    rate: number | null;
    rf_daily_return: number | null;
}

export interface CorrelationMatrixPayload {
    status: "sufficient" | "partial" | "unavailable";
    n_obs: number;
    assets_included: string[];
    assets_excluded: { ticker: string; reason: string }[];
    matrix: Record<string, Record<string, number>>;
    reasons?: string[];
}

export interface CorporateActionEvent {
    date: string;
    ticker: string;
    dividend?: number | null;
    split_ratio?: number | null;
}

export interface DataContractsPayload {
    [key: string]: {
        version: string;
    };
}

export interface DiagnosticSignal {
    key: string;
    /** Alternative check identifier used by some components */
    check?: string;
    category: "risk" | "performance" | "data" | "structure";
    severity: "low" | "medium" | "high" | "critical";
    summary: string;
    /** Human-readable diagnostic message */
    message?: string;
    evidence: string[];
    metrics_used: string[];
    as_of: string | null;
    confidence: number;
    suggested_action?: string;
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

export type MetricCoverageStatus = "sufficient" | "insufficient" | "unknown" | "available_low_coverage" | "unavailable";

export interface CoverageSourceStatus {
    status: MetricCoverageStatus;
    reason_codes: string[];
}

export interface CoverageSummaryDetailed {
    as_of: string | null;
    status: MetricCoverageStatus;
    score: number;
    policy: CoveragePolicy;
    required: {
        tickers: string[];
        history_days_needed: number;
    };
    per_ticker: Record<string, CoveragePerTicker>;
    aggregate: CoverageAggregate;
    coverage?: Record<string, CoverageSourceStatus>;
    metric_status?: Record<string, MetricCoverageStatus>;
    metric_reasons?: Record<string, string[]>;
    reason_codes: string[];
    contract_version?: string;
    version?: string;
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
    diagnostics?: DiagnosticSignal[];
    correlation_matrix?: CorrelationMatrixPayload | null;
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
    risk_free_series?: RiskFreeSeriesPoint[];
    corporate_actions?: CorporateActionEvent[];
    data_contracts?: DataContractsPayload | null;
    portfolio: PortfolioMeta | null;
    definitions?: DefinitionsRegistry;
}

export interface RunMetricsResponse {
    manifest: RunManifest;
    coverage_summary?: CoverageSummaryDetailed | null;
    diagnostics?: DiagnosticSignal[];
    correlation_matrix?: CorrelationMatrixPayload | null;
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
    risk_free_series?: RiskFreeSeriesPoint[];
    corporate_actions?: CorporateActionEvent[];
    data_contracts?: DataContractsPayload | null;
}

export interface OpsHealthResponse {
    status: string;
    runtime: {
        uptime_seconds: number;
        started_at: string;
        now: string;
        rss_mb: number;
    };
    database: {
        status: string;
        backend: string;
    };
    cache: Record<string, unknown>;
    rate_limit: {
        enabled: boolean;
        limit_per_window: number;
        window_seconds: number;
    };
    latest_run: {
        run_id: string;
        status: string;
        timestamp: string | null;
    } | null;
}

export interface PortfolioResponse {
    portfolio: PortfolioMeta;
    holdings: Holding[];
}
