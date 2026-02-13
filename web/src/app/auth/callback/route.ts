import { createServerClient } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";

function resolveNextPath(nextParam: string | null): string {
  if (!nextParam || !nextParam.startsWith("/")) return "/onboarding";
  return nextParam;
}

export async function GET(request: NextRequest) {
  const requestUrl = new URL(request.url);
  const code = requestUrl.searchParams.get("code");
  const nextPath = resolveNextPath(requestUrl.searchParams.get("next"));

  const redirectUrl = new URL(nextPath, requestUrl.origin);
  const loginUrl = new URL("/login", requestUrl.origin);

  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !anonKey) {
    loginUrl.searchParams.set("error", "Supabase auth is not configured in this environment.");
    return NextResponse.redirect(loginUrl);
  }

  let response = NextResponse.redirect(redirectUrl);
  const supabase = createServerClient(url, anonKey, {
    cookies: {
      getAll() {
        return request.cookies.getAll();
      },
      setAll(
        cookiesToSet: Array<{
          name: string;
          value: string;
          options: Parameters<typeof response.cookies.set>[2];
        }>
      ) {
        response = NextResponse.redirect(redirectUrl);
        for (const { name, value, options } of cookiesToSet) {
          request.cookies.set(name, value);
          response.cookies.set(name, value, options);
        }
      },
    },
  });

  if (code) {
    const { error } = await supabase.auth.exchangeCodeForSession(code);
    if (error) {
      loginUrl.searchParams.set("error", "Could not verify email. Please sign in manually.");
      return NextResponse.redirect(loginUrl);
    }
  }

  return response;
}
