const API_BASE_URL =
    process.env.NEXUS_API_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://localhost:8000";

export async function fetchBackend<T>(path: string): Promise<T> {
    const res = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store" });
    if (!res.ok) {
        throw new Error(`Backend request failed: ${res.status}`);
    }
    return res.json() as Promise<T>;
}
