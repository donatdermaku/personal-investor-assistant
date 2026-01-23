import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/nexus/Sidebar";
import { ContextPanel } from "@/components/nexus/ContextPanel";
import { NexusProvider } from "@/components/nexus/NexusProvider";
import { TopBar } from "@/components/nexus/TopBar";
import { RunCreationModal } from "@/components/nexus/RunCreationModal";

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
        <NexusProvider>
          {/* Zone 1: Sidebar (Fixed Left) */}
          <Sidebar />

          {/* Zone 2: Main Content (Fluid Center) */}
          <main className="min-h-screen p-6 lg:ml-64 lg:mr-80 lg:p-8 pb-24 lg:pb-8">
            <div className="max-w-5xl mx-auto">
              <TopBar />
              {children}
            </div>
          </main>

          {/* Zone 3: Context Panel (Fixed Right on Large screens) */}
          <ContextPanel />

          <RunCreationModal />
        </NexusProvider>
      </body>
    </html>
  );
}
