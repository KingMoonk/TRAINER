"use client";

// Light / dark / system (next-themes puts the `dark` class on <html>) and the accent (a data-theme attribute on
// <html>). Both are read from the DOM, so a component re-renders when either changes, wherever it was changed from.

import { ThemeProvider } from "next-themes";
import { useMemo, useSyncExternalStore, type ReactNode } from "react";
import { ACCENT_KEY, ACCENTS, type Accent } from "@/lib/theme";

export function AppearanceProvider({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
      {children}
    </ThemeProvider>
  );
}

/** Calls `notify` whenever <html>'s class (light/dark) or data-theme (accent) changes. */
function watchRoot(notify: () => void): () => void {
  const observer = new MutationObserver(notify);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class", "data-theme"] });
  return () => observer.disconnect();
}

function currentAccent(): Accent {
  const id = document.documentElement.getAttribute("data-theme");
  return ACCENTS.find((accent) => accent === id) ?? "neutral";
}

export function useAccent(): [Accent, (accent: Accent) => void] {
  const accent = useSyncExternalStore(watchRoot, currentAccent, () => "neutral" as Accent);
  const setAccent = (next: Accent) => {
    if (next === "neutral") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", next);
    try {
      if (next === "neutral") localStorage.removeItem(ACCENT_KEY);
      else localStorage.setItem(ACCENT_KEY, next);
    } catch {
      // Storage can be blocked; the accent then lasts until the page is closed.
    }
  };
  return [accent, setAccent];
}

let probe: CanvasRenderingContext2D | null | undefined;

/** A token as a colour string for code that cannot read CSS variables (three.js): `#rrggbb` (alpha dropped), or "" if unset. */
function resolveColor(name: string): string {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  if (!raw) return "";
  probe ??= document.createElement("canvas").getContext("2d", { willReadFrequently: true });
  if (!probe) return raw;
  // The canvas parses any CSS colour (oklch, color-mix, ...) and hands back plain RGB.
  probe.clearRect(0, 0, 1, 1);
  probe.fillStyle = "black";
  probe.fillStyle = raw;
  probe.fillRect(0, 0, 1, 1);
  const [r, g, b] = probe.getImageData(0, 0, 1, 1).data;
  return `#${[r, g, b].map((channel) => channel.toString(16).padStart(2, "0")).join("")}`;
}

/**
 * The current value of a theme token, e.g. `useThemeColor("--scene-background")`, resolved to a colour string that
 * three.js and canvas code accept. It updates when the mode or the accent changes. "" before the first client render.
 * CSS and inline styles do not need this: they use `var(--token)` directly.
 */
export function useThemeColor(name: string): string {
  return useSyncExternalStore(watchRoot, () => resolveColor(name), () => "");
}

/**
 * Several tokens at once, in the order asked: `useThemeColors(["--ramp-1", "--ramp-2"])`. Each is "" before the first
 * client render. Pass a module-level array (the hook compares the names, not the array).
 */
export function useThemeColors(names: readonly string[]): string[] {
  const key = names.join("|");
  const joined = useSyncExternalStore(watchRoot, () => key.split("|").map(resolveColor).join("|"), () => "");
  return useMemo(() => (joined ? joined.split("|") : key.split("|").map(() => "")), [joined, key]);
}
