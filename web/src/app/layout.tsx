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
      <body className="antialiased bg-[var(--color-nexus-bg)] text-[var(--color-nexus-text-primary)] selection:bg-[var(--color-nexus-primary)] selection:text-black overflow-x-hidden min-h-screen">
        <NexusProvider>
          <AmbientBackground />
          <div className="app-shell relative z-10">
            <Sidebar />
            <main className="app-content">
              <div className="app-content-inner">
                <TopBar />
                {children}
              </div>
            </main>
            <MissionControl />
          </div>
        </NexusProvider>
      </body>
    </html>
  );
}
