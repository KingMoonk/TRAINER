import type { Metadata } from "next";
import "./globals.css";
import { InlineScript } from "@/components/inline-script";
import { Providers } from "@/components/providers";
import { Sidebar } from "@/components/sidebar";
import { StatusStrip } from "@/components/status-strip";
import { Toaster } from "@/components/ui/sonner";
import { ShellEvents } from "@/lib/events";
import { ACCENT_SCRIPT } from "@/lib/theme";

export const metadata: Metadata = {
  title: "Groundwork Studio",
  description: "Create datasets and train your own language model from random weights.",
};

// Dark is the default (so the prerendered page is dark even before any script runs). next-themes puts the mode class on <html> and ACCENT_SCRIPT the accent attribute, both before
// first paint, so <html> differs from the prerendered markup and needs suppressHydrationWarning.
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark h-full antialiased" suppressHydrationWarning>
      <head>
        <InlineScript code={ACCENT_SCRIPT} />
      </head>
      <body className="flex min-h-dvh flex-col md:flex-row">
        <Providers>
          <ShellEvents>
            <a href="#studio-content" className="sr-only z-50 rounded-lg bg-primary px-4 py-2 text-primary-foreground focus:not-sr-only focus:fixed focus:left-4 focus:top-4">Skip to content</a>
            <Sidebar />
            <div id="studio-content" tabIndex={-1} className="flex min-w-0 flex-1 flex-col outline-none">
              <StatusStrip />
              {children}
            </div>
            <Toaster />
          </ShellEvents>
        </Providers>
      </body>
    </html>
  );
}
