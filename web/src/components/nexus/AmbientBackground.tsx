"use client";

import React from "react";

export function AmbientBackground() {
    return (
        <div className="fixed inset-0 z-[-1] overflow-hidden bg-[var(--color-nexus-bg)] pointer-events-none">
            {/* Subtle Grid Pattern */}
            <div className="absolute inset-0 bg-grid-pattern opacity-20" />

            {/* Ambient Orbs */}
            <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-[var(--color-nexus-primary)] rounded-full blur-[120px] opacity-10 animate-aurora mix-blend-screen" />
            <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-[var(--color-nexus-accent)] rounded-full blur-[120px] opacity-10 animate-aurora mix-blend-screen delay-1000" />
            <div className="absolute top-[20%] left-[30%] w-[40%] h-[40%] bg-[var(--color-nexus-secondary)] rounded-full blur-[100px] opacity-5 animate-pulse-slow mix-blend-screen delay-2000" />

            {/* Vignette */}
            <div className="absolute inset-0 bg-radial-gradient from-transparent to-[var(--color-nexus-bg)]/80" />
        </div>
    );
}
