import { NextResponse } from "next/server";
import { fetchBackend } from "@/lib/api-server";
import { MOCK_HOLDINGS, MOCK_PORTFOLIO } from "@/lib/mock-data";

export async function GET() {
    try {
        const data = await fetchBackend("/portfolio/default");
        return NextResponse.json(data);
    } catch {
        return NextResponse.json({ portfolio: MOCK_PORTFOLIO, holdings: MOCK_HOLDINGS });
    }
}
