import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/nexus/Sidebar";
import { ContextPanel } from "@/components/nexus/ContextPanel";

export const metadata: Metadata = {
  title: "Nexus Analytics Platform",
  description: "Professional Investment Analytics",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased bg-[#F3F4F6] text-[#0F172A]">
        {/* Zone 1: Sidebar (Fixed Left) */}
        <Sidebar />

        {/* Zone 2: Main Content (Fluid Center) */}
        <main className="ml-64 mr-0 lg:mr-80 min-h-screen p-8">
          <div className="max-w-5xl mx-auto">
            {children}
          </div>
        </main>

        {/* Zone 3: Context Panel (Fixed Right on Large screens) */}
        <ContextPanel />
      </body>
    </html>
  );
}
