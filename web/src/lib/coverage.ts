import type { CoverageSummary, CoverageSummaryDetailed, RunManifest } from "@/types/nexus";

export type CoverageStatus = "full" | "partial" | "insufficient" | "unknown" | "available_low_coverage";

function isDetailedSummary(summary: unknown): summary is CoverageSummaryDetailed {
    return Boolean(summary && typeof summary === "object" && "status" in summary && "score" in summary);
}

function legacyCoveragePercent(summary: Record<string, CoverageSummary>): number | null {
    let covered = 0;
    let total = 0;
    Object.values(summary).forEach((item) => {
        covered += item.covered || 0;
        total += item.total || 0;
    });
    if (total <= 0) return null;
    return covered / total;
}

export function coveragePercent(
    manifest?: RunManifest | null,
    detailedSummary?: CoverageSummaryDetailed | null
): number | null {
    if (detailedSummary) {
        return typeof detailedSummary.score === "number" ? detailedSummary.score : null;
    }
    const summary = manifest?.coverage_summary;
    if (!summary || Object.keys(summary).length === 0) return null;
    if (isDetailedSummary(summary)) {
        return typeof summary.score === "number" ? summary.score : null;
    }
    return legacyCoveragePercent(summary as Record<string, CoverageSummary>);
}

export function coverageStatus(
    manifest?: RunManifest | null,
    detailedSummary?: CoverageSummaryDetailed | null
): CoverageStatus {
    const summary = detailedSummary ?? manifest?.coverage_summary ?? null;
    if (summary && isDetailedSummary(summary)) {
        if (summary.status === "unknown") return "unknown";
        const score = typeof summary.score === "number" ? summary.score : 0;
        if (summary.status === "insufficient") return "insufficient";
        if (score >= 0.95) return "full";
        if (score >= 0.75) return "partial";
        return "insufficient";
    }
    const percent = coveragePercent(manifest, detailedSummary);
    if (percent === null) return "unknown";
    if (percent >= 0.95) return "full";
    if (percent >= 0.75) return "partial";
    return "insufficient";
}

export function coverageLabel(status: CoverageStatus): string {
    switch (status) {
        case "full":
            return "Full coverage";
        case "partial":
            return "Partial coverage";
        case "unknown":
            return "Coverage unknown";
        default:
            return "Insufficient data";
    }
}
