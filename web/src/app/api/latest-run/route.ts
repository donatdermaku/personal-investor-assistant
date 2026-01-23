import { NextResponse } from "next/server";
import { fetchBackend } from "@/lib/api-server";
import { MOCK_MANIFEST } from "@/lib/mock-data";

export async function GET() {
    try {
        const data = await fetchBackend("/latest-run");
        return NextResponse.json(data);
    } catch {
        return NextResponse.json(MOCK_MANIFEST);
    }
}
