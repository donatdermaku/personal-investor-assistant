"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createPortfolio, createRun } from "@/lib/api";
import { parseCsvHeaders, validateOnboardingCsvHeaders } from "@/lib/onboarding";

export default function OnboardingPage() {
  const router = useRouter();
  const [portfolioName, setPortfolioName] = useState("Main Portfolio");
  const [currency, setCurrency] = useState("USD");
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleFile = async (picked: File | null) => {
    setFile(picked);
    setError(null);
    if (!picked) return;
    const text = await picked.text();
    const missing = validateOnboardingCsvHeaders(parseCsvHeaders(text));
    if (missing.length > 0) {
      setError(`CSV missing required columns: ${missing.join(", ")}`);
      return;
    }
    if (picked.size > 4 * 1024 * 1024) {
      setWarning("Large file detected (>4MB). Upload may take longer.");
    } else {
      setWarning(null);
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    if (!file) {
      setError("Please upload a CSV before continuing.");
      return;
    }
    setLoading(true);
    try {
      const created = await createPortfolio({ name: portfolioName, currency });
      const run = await createRun({
        runType: "uploaded",
        portfolioId: String(created.portfolio.id),
        file,
      });
      if (run.warnings?.failed_tickers?.message) {
        setWarning(run.warnings.failed_tickers.message);
      }
      router.push("/overview");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Onboarding failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <h1 className="mb-2 text-3xl font-semibold">Onboarding</h1>
      <p className="mb-8 text-sm text-[var(--color-nexus-text-secondary)]">
        Create your first portfolio and upload a transaction CSV to generate analytics.
      </p>

      <form onSubmit={handleSubmit} className="space-y-5 border border-[var(--color-nexus-border)] bg-[var(--color-nexus-surface)] p-6">
        <div>
          <label className="mb-1 block text-xs uppercase tracking-wide text-[var(--color-nexus-text-secondary)]">
            Portfolio Name
          </label>
          <input
            required
            maxLength={80}
            value={portfolioName}
            onChange={(event) => setPortfolioName(event.target.value)}
            className="w-full border border-[var(--color-nexus-border)] bg-black/20 px-3 py-2 text-sm"
          />
        </div>

        <div>
          <label className="mb-1 block text-xs uppercase tracking-wide text-[var(--color-nexus-text-secondary)]">
            Base Currency
          </label>
          <input
            required
            maxLength={3}
            value={currency}
            onChange={(event) => setCurrency(event.target.value.toUpperCase())}
            className="w-full border border-[var(--color-nexus-border)] bg-black/20 px-3 py-2 text-sm"
          />
        </div>

        <div>
          <label className="mb-1 block text-xs uppercase tracking-wide text-[var(--color-nexus-text-secondary)]">
            Transactions CSV
          </label>
          <input
            required
            type="file"
            accept=".csv"
            onChange={(event) => {
              const picked = event.target.files?.[0] ?? null;
              void handleFile(picked);
            }}
            className="w-full text-sm file:mr-3 file:border file:border-[var(--color-nexus-border)] file:bg-transparent file:px-3 file:py-1"
          />
          <p className="mt-2 text-xs text-[var(--color-nexus-text-muted)]">
            Required columns: date, ticker, action, quantity (or shares), price.
          </p>
        </div>

        {warning && <p className="text-xs text-[var(--color-nexus-warning)]">{warning}</p>}
        {error && <p className="text-xs text-[var(--color-nexus-danger)]">{error}</p>}

        <button
          type="submit"
          disabled={loading}
          className="w-full border border-[var(--color-nexus-primary)] px-4 py-2 text-sm font-semibold disabled:opacity-50"
        >
          {loading ? "Creating portfolio and run..." : "Create Portfolio and Analyze"}
        </button>
      </form>
    </main>
  );
}
