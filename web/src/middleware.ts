import { NextResponse, type NextRequest } from "next/server";
import { createServerClient } from "@supabase/ssr";
import { updateSession } from "@/lib/supabase/middleware";

const PROTECTED_PREFIXES = ["/overview", "/performance", "/holdings", "/risk", "/operations"];

function isProtectedPath(pathname: string): boolean {
  return PROTECTED_PREFIXES.some((prefix) => pathname.startsWith(prefix));
}

export async function middleware(request: NextRequest) {
  const baseResponse = await updateSession(request);
  if (!isProtectedPath(request.nextUrl.pathname)) {
    return baseResponse;
  }

  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !anonKey) {
    return baseResponse;
  }

  const supabase = createServerClient(url, anonKey, {
    cookies: {
      getAll() {
        return request.cookies.getAll();
      },
      setAll(
        cookiesToSet: Array<{
          name: string;
          value: string;
          options: Parameters<typeof baseResponse.cookies.set>[2];
        }>
      ) {
        for (const { name, value, options } of cookiesToSet) {
          request.cookies.set(name, value);
          baseResponse.cookies.set(name, value, options);
        }
      },
    },
  });

  const { data } = await supabase.auth.getUser();
  if (!data.user) {
    const loginUrl = request.nextUrl.clone();
    loginUrl.pathname = "/login";
    loginUrl.searchParams.set("next", request.nextUrl.pathname);
    return NextResponse.redirect(loginUrl);
  }
  return baseResponse;
}

export const config = {
  matcher: ["/overview/:path*", "/performance/:path*", "/holdings/:path*", "/risk/:path*", "/operations/:path*"],
};
