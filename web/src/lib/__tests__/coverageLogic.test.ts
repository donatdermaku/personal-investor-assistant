import { describe, expect, it } from "vitest";

import { shouldHideKpiValue } from "@/lib/coverageLogic";

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
