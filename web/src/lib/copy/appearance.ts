// The Settings page (Appearance is its first section)'s strings. The colours are tokens in app/globals.css; the ids are in lib/theme.ts.
import type { Accent, Mode } from "@/lib/theme";

export const APPEARANCE = {
  h1: "Settings",
  lead: "Preferences for this computer.",
  settingsLabel: "Settings",
  section: "Appearance",
  sectionLead: "Pick light or dark, and an accent colour. Kept in this browser.",
  modeTitle: "Mode",
  modeDesc: "System follows your computer's light or dark setting.",
  modes: { light: "Light", dark: "Dark", system: "System" } satisfies Record<Mode, string>,
  accentTitle: "Accent",
  accentDesc: "The colour of buttons, links, highlights and charts. It works in both modes.",
  accents: { neutral: "Neutral", ocean: "Ocean", ember: "Ember", forest: "Forest" } satisfies Record<Accent, string>,
};
