import { describe, expect, it } from "vitest";

import { shouldHideKpiValue, getKpiBadge } from "@/lib/coverageLogic";

describe("shouldHideKpiValue", () => {
    it("shows KPI when coverage summary is missing", () => {
        expect(shouldHideKpiValue("twr", null)).toBe(false);
    });

    it("hides only metrics whose dependencies are insufficient", () => {
        const coverageSummary = {
            metric_status: {
                twr: "sufficient",
                sharpe: "insufficient",
                allocation_effect: "insufficient",
                max_drawdown: "sufficient",
            },
            metric_reasons: {
                sharpe: ["RF_MISSING"],
                allocation_effect: ["BENCHMARK_INSUFFICIENT"],
            },
        };

        expect(shouldHideKpiValue("twr", coverageSummary as any)).toBe(false);
        expect(shouldHideKpiValue("sharpe", coverageSummary as any)).toBe(true);
        expect(shouldHideKpiValue("allocation_effect", coverageSummary as any)).toBe(true);
        expect(shouldHideKpiValue("max_drawdown", coverageSummary as any)).toBe(false);
    });

    it("hides KPIs when prices are insufficient", () => {
        const coverageSummary = {
            metric_status: {
                twr: "insufficient",
                portfolio_value: "insufficient",
            },
            metric_reasons: {
                twr: ["PRICE_INSUFFICIENT"],
                portfolio_value: ["PRICE_INSUFFICIENT"],
            },
        };

        expect(shouldHideKpiValue("twr", coverageSummary as any)).toBe(true);
        expect(shouldHideKpiValue("portfolio_value", coverageSummary as any)).toBe(true);
    });
});

describe('getKpiBadge', () => {
    it('returns INSUFFICIENT when hidden', () => {
        const summary: any = { metric_status: { sharpe: "insufficient" } };
        expect(getKpiBadge('sharpe', summary)).toBe('INSUFFICIENT');
    });

    it('returns WARNING when available_low_coverage', () => {
        const summary: any = { metric_status: { sharpe: "available_low_coverage" } };
        expect(getKpiBadge('sharpe', summary)).toBe('WARNING');
    });

    it('returns null when sufficient', () => {
        const summary: any = { metric_status: { sharpe: "sufficient" } };
        expect(getKpiBadge('sharpe', summary)).toBe(null);
    });
});
