"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { getSupabaseBrowserClient } from "@/lib/supabase/browser";

export default function HomePage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const supabaseReady = useMemo(
    () => Boolean(process.env.NEXT_PUBLIC_SUPABASE_URL && process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY),
    []
  );

  const handleSignup = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setNotice(null);
    const supabase = getSupabaseBrowserClient();
    if (!supabase) {
      setError("Signup is unavailable because Supabase env vars are not configured.");
      return;
    }
    setLoading(true);
    const { data, error: signUpError } = await supabase.auth.signUp({ email, password });
    setLoading(false);
    if (signUpError) {
      setError(signUpError.message);
      return;
    }
    if (data.session) {
      router.push("/onboarding");
      return;
    }
    setNotice("Check your email to confirm your account, then sign in to start onboarding.");
  };

  return (
    <main className="mx-auto min-h-screen max-w-6xl px-6 py-16">
      <section className="grid gap-10 lg:grid-cols-2 lg:items-center">
        <div>
          <p className="mb-3 text-xs uppercase tracking-[0.2em] text-[var(--color-nexus-primary)]">Nexus Beta</p>
          <h1 className="mb-4 text-4xl font-semibold tracking-tight text-[var(--color-nexus-text-primary)] md:text-5xl">
            Sign up for beta access
          </h1>
          <p className="max-w-xl text-base text-[var(--color-nexus-text-secondary)]">
            Upload your portfolio transactions and get risk, attribution, concentration, and benchmark insights in minutes.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => router.push("/login")}
              className="border border-[var(--color-nexus-border)] px-4 py-2 text-sm hover:border-[var(--color-nexus-primary)]"
            >
              I already have an account
            </button>
            <button
              type="button"
              onClick={() => router.push("/overview")}
              className="border border-[var(--color-nexus-border)] px-4 py-2 text-sm hover:border-[var(--color-nexus-primary)]"
            >
              Open dashboard
            </button>
          </div>
        </div>

        <div className="border border-[var(--color-nexus-border)] bg-[var(--color-nexus-surface)] p-6">
          <h2 className="mb-4 text-lg font-semibold">Create Beta Account</h2>
          <form onSubmit={handleSignup} className="space-y-3">
            <input
              type="email"
              required
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="Email"
              className="w-full border border-[var(--color-nexus-border)] bg-black/20 px-3 py-2 text-sm"
            />
            <input
              type="password"
              required
              minLength={8}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Password (min 8 chars)"
              className="w-full border border-[var(--color-nexus-border)] bg-black/20 px-3 py-2 text-sm"
            />
            {error && <p className="text-xs text-[var(--color-nexus-danger)]">{error}</p>}
            {notice && <p className="text-xs text-[var(--color-nexus-success)]">{notice}</p>}
            <button
              type="submit"
              disabled={loading || !supabaseReady}
              className="w-full border border-[var(--color-nexus-primary)] px-3 py-2 text-sm font-medium disabled:opacity-50"
            >
              {loading ? "Creating account..." : "Sign up for beta"}
            </button>
          </form>
          {!supabaseReady && (
            <p className="mt-3 text-xs text-[var(--color-nexus-warning)]">
              Supabase browser env vars are missing in this environment.
            </p>
          )}
        </div>
      </section>
    </main>
  );
}
