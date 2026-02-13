import { describe, expect, it } from "vitest";
import { parseCsvHeaders, validateOnboardingCsvHeaders } from "@/lib/onboarding";

describe("onboarding CSV helpers", () => {
  it("parses and normalizes csv headers", () => {
    const headers = parseCsvHeaders('"Date",Ticker, ACTION ,Quantity,Price\n2026-01-01,AAPL,BUY,1,100');
    expect(headers).toEqual(["date", "ticker", "action", "quantity", "price"]);
  });

  it("returns missing required headers", () => {
    const missing = validateOnboardingCsvHeaders(["date", "ticker", "price"]);
    expect(missing).toEqual(["action", "quantity (or shares)"]);
  });

  it("accepts shares column as quantity alias", () => {
    const missing = validateOnboardingCsvHeaders(["date", "ticker", "action", "price", "shares"]);
    expect(missing).toEqual([]);
  });
});
