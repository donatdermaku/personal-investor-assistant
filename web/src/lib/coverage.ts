import type { RunManifest } from "@/types/nexus";

export type CoverageStatus = "full" | "partial" | "insufficient" | "unknown";

export function coveragePercent(manifest?: RunManifest | null): number | null {
    const summary = manifest?.coverage_summary;
    if (!summary || Object.keys(summary).length === 0) return null;
    let covered = 0;
    let total = 0;
    Object.values(summary).forEach((item) => {
        covered += item.covered || 0;
        total += item.total || 0;
    });
    if (total <= 0) return null;
    return covered / total;
}

export function coverageStatus(manifest?: RunManifest | null): CoverageStatus {
    const percent = coveragePercent(manifest);
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
