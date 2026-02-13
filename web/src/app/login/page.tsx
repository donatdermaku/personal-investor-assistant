"use client";

import { Suspense, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { getSupabaseBrowserClient } from "@/lib/supabase/browser";

function LoginPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const nextPath = useMemo(() => searchParams.get("next") || "/overview", [searchParams]);
  const callbackError = useMemo(() => searchParams.get("error"), [searchParams]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    const supabase = getSupabaseBrowserClient();
    if (!supabase) {
      setError("Supabase env vars are missing.");
      return;
    }
    setLoading(true);
    try {
      const { data, error: signInError } = await supabase.auth.signInWithPassword({ email, password });
      if (signInError) {
        setError(signInError.message);
        return;
      }
      if (!data.session) {
        setError("Sign-in did not return a session. Please verify your email and try again.");
        return;
      }

      const safeNextPath = nextPath.startsWith("/") ? nextPath : "/overview";
      router.replace(safeNextPath);
      router.refresh();
      // Fallback for cases where middleware/session state race blocks client navigation.
      window.location.assign(safeNextPath);
    } catch {
      setError("Sign-in request failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-md border border-[var(--color-nexus-border)] bg-[var(--color-nexus-surface)] p-6">
      <h1 className="mb-4 text-xl font-semibold">Login</h1>
      <form onSubmit={handleSubmit} className="space-y-3">
        <input
          type="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          className="w-full border border-[var(--color-nexus-border)] bg-black/20 px-3 py-2 text-sm"
        />
        <input
          type="password"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          className="w-full border border-[var(--color-nexus-border)] bg-black/20 px-3 py-2 text-sm"
        />
        {(error || callbackError) && (
          <p className="text-xs text-[var(--color-nexus-danger)]">{error || callbackError}</p>
        )}
        <button
          type="submit"
          disabled={loading}
          className="w-full border border-[var(--color-nexus-primary)] px-3 py-2 text-sm font-medium disabled:opacity-50"
        >
          {loading ? "Signing in..." : "Sign in"}
        </button>
      </form>
      <button
        type="button"
        onClick={() => router.push("/")}
        className="mt-3 w-full border border-[var(--color-nexus-border)] px-3 py-2 text-sm"
      >
        Create new account
      </button>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={<div className="mx-auto max-w-md p-6 text-sm">Loading...</div>}>
      <LoginPageContent />
    </Suspense>
  );
}
