export interface RunManifest {
    run_id: string;
    timestamp: string | null;
    input_hash: string | null;
    data_hash: string | null;
    status?: "running" | "completed" | "failed";
    code_version?: string | null;
    coverage_summary?: Record<string, CoverageSummary>;
    meta?: Record<string, unknown>;
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

export interface CoverageSummary {
    total: number;
    covered: number;
    last_date?: string | null;
    missing_count?: number;
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
    summary: PortfolioSummary;
    equity_curve: TimeSeriesPoint[];
    performance: PerformancePoint[];
    monthly_returns: MonthlyReturnPoint[];
    holdings: Holding[];
    risk: RiskMetrics;
    portfolio: PortfolioMeta | null;
    definitions?: DefinitionsRegistry;
}

export interface RunMetricsResponse {
    manifest: RunManifest;
    summary: PortfolioSummary;
    equity_curve: TimeSeriesPoint[];
    performance: PerformancePoint[];
    monthly_returns: MonthlyReturnPoint[];
    risk: RiskMetrics;
}

export interface PortfolioResponse {
    portfolio: PortfolioMeta;
    holdings: Holding[];
}
