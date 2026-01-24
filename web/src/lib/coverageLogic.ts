import type { CoverageSummaryDetailed, MetricCoverageStatus } from "@/types/nexus";

export function getMetricStatus(
    kpiKey: string,
    coverageSummary?: CoverageSummaryDetailed | null
): MetricCoverageStatus {
    const status = coverageSummary?.metric_status?.[kpiKey];
    if (status === "sufficient" || status === "insufficient" || status === "unknown") {
        return status;
    }
    return "unknown";
}

export function getMetricReasons(
    kpiKey: string,
    coverageSummary?: CoverageSummaryDetailed | null
): string[] {
    return coverageSummary?.metric_reasons?.[kpiKey] ?? [];
}

export function shouldHideKpiValue(
    kpiKey: string,
    coverageSummary?: CoverageSummaryDetailed | null
): boolean {
    return getMetricStatus(kpiKey, coverageSummary) === "insufficient";
}

export function getKpiBadge(
    kpiKey: string,
    coverageSummary?: CoverageSummaryDetailed | null
): "INSUFFICIENT" | null {
    return shouldHideKpiValue(kpiKey, coverageSummary) ? "INSUFFICIENT" : null;
}
