export interface RunManifest {
    run_id: string;
    timestamp: string;
    input_hash: string;
    data_hash: string;
    status: "running" | "completed" | "failed";
}

export interface PortfolioSummary {
    source: string;
    twr: number | null;
    mwr: number | null; // Personal return (IRR)
    final_value: number | null;
    last_date: string | null;
    max_drawdown: number | null;
    errors: string[];
}

export interface TimeSeriesPoint {
    date: string;
    value: number;
}

export interface Holding {
    ticker: string;
    shares?: number;
    price?: number;
    value?: number;
    weight?: number; // 0-1
    sector?: string;
}

export interface RiskMetrics {
    var_95: number | null;
    cvar_95: number | null;
    volatility: number | null;
    sharpe: number | null;
}

export interface NexusState {
    manifest: RunManifest;
    summary: PortfolioSummary;
    equity_curve: TimeSeriesPoint[];
    holdings: Holding[];
    risk: RiskMetrics;
}
