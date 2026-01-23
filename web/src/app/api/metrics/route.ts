import { NextResponse } from "next/server";
import { fetchBackend } from "@/lib/api-server";
import { MOCK_METRICS } from "@/lib/mock-data";

export async function GET() {
    try {
        const latest = await fetchBackend<{ run_id: string }>("/latest-run");
        const runId = latest?.run_id;
        if (!runId) {
            return NextResponse.json(MOCK_METRICS);
        }
        const data = await fetchBackend(`/run/${runId}`);
        return NextResponse.json(data);
    } catch {
        return NextResponse.json(MOCK_METRICS);
    }
}
