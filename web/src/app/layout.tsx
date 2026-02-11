import type { Metadata } from "next";
import { Outfit, Space_Grotesk } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/nexus/Sidebar";
import { MissionControl } from "@/components/nexus/MissionControl";
import { NexusProvider } from "@/components/nexus/NexusProvider";
import { TopBar } from "@/components/nexus/TopBar";
import { AmbientBackground } from "@/components/nexus/AmbientBackground";

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-outfit",
  display: "swap",
});

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
  display: "swap",
});

export const metadata: Metadata = {
  title: "NEXUS | Private Perspective",
  description: "Advanced Investment Intelligence",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${outfit.variable} ${spaceGrotesk.variable}`}>
      <body className="antialiased bg-[var(--color-nexus-bg)] text-[var(--color-nexus-text-primary)] selection:bg-[var(--color-nexus-primary)] selection:text-black overflow-hidden h-screen">
        <NexusProvider>
          <AmbientBackground />

          <div className="flex h-screen overflow-hidden">
            {/* Zone 1: Sidebar (Fixed Left) */}
            <Sidebar />

            {/* Zone 2: Main Content (Fluid Center) */}
            <main className="flex-1 overflow-y-auto relative z-10">
              <div className="max-w-7xl mx-auto p-6 lg:p-8 pb-24 lg:pb-8">
                <TopBar />
                {children}
              </div>
            </main>

            {/* Zone 3: Mission Control (Fixed Right) */}
            <MissionControl />
          </div>

        </NexusProvider>
      </body>
    </html>
  );
}
