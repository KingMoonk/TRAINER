import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "@/components/providers";
import { Sidebar } from "@/components/sidebar";
import { StatusStrip } from "@/components/status-strip";
import { Toaster } from "@/components/ui/sonner";
import { ShellEvents } from "@/lib/events";

export const metadata: Metadata = {
  title: "Groundwork Studio",
  description: "Create datasets and train your own language model from random weights.",
};

// Dark is the default and the only theme for now: the `dark` class on <html> is what globals.css keys on.
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark h-full antialiased">
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
