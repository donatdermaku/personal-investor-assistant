import type { CoverageSummaryDetailed, MetricCoverageStatus } from "@/types/nexus";

export function getMetricStatus(
    kpiKey: string,
    coverageSummary?: CoverageSummaryDetailed | null
): MetricCoverageStatus {
    const status = coverageSummary?.metric_status?.[kpiKey];
    if (status === "sufficient" || status === "insufficient" || status === "unknown" || status === "available_low_coverage" || status === "unavailable") {
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
    const status = getMetricStatus(kpiKey, coverageSummary);
    // Hide ONLY if unavailable, or if purely insufficient (legacy/missing block)
    // "available_low_coverage" should NOT hide.
    return status === "unavailable" || status === "insufficient";
}

export function getKpiBadge(
    kpiKey: string,
    coverageSummary?: CoverageSummaryDetailed | null
): "INSUFFICIENT" | "WARNING" | null {
    const status = getMetricStatus(kpiKey, coverageSummary);
    if (status === "available_low_coverage") return "WARNING";
    if (shouldHideKpiValue(kpiKey, coverageSummary)) return "INSUFFICIENT";
    return null;
}
