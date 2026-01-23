import { NextResponse } from "next/server";
import { fetchBackend } from "@/lib/api-server";
import type { PortfolioResponse } from "@/types/nexus";
import { MOCK_HOLDINGS } from "@/lib/mock-data";

export async function GET() {
    try {
        const data = await fetchBackend<PortfolioResponse>("/portfolio/default");
        return NextResponse.json(data.holdings || []);
    } catch {
        return NextResponse.json(MOCK_HOLDINGS);
    }
}
