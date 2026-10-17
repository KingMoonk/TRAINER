// Appearance constants shared by the layout (a server component) and components/theme.tsx. The colours themselves are
// tokens in app/globals.css; this file only names the choices. Add an accent there, then its id here and its label in
// lib/copy/appearance.ts.

export const MODES = ["light", "dark", "system"] as const;
export type Mode = (typeof MODES)[number];

/** "neutral" is no data-theme attribute at all: today's look. */
export const ACCENTS = ["neutral", "ocean", "ember", "forest"] as const;
export type Accent = (typeof ACCENTS)[number];

export const ACCENT_KEY = "groundwork-accent";

/**
 * Runs in <head> before first paint, so a saved accent is on <html> before anything is drawn (the app is a static
 * export, so there is no server to put it there). next-themes does the same for light/dark/system.
 */
export const ACCENT_SCRIPT = `try{var a=localStorage.getItem(${JSON.stringify(ACCENT_KEY)});if(${JSON.stringify(ACCENTS.filter((id) => id !== "neutral"))}.indexOf(a)>-1)document.documentElement.setAttribute("data-theme",a)}catch(e){}`;
