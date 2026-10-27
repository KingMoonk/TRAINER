// The Settings page (Appearance is its first section)'s strings. The colours are tokens in app/globals.css; the ids are in lib/theme.ts.
import type { Accent, Mode } from "@/lib/theme";

export const APPEARANCE = {
  h1: "Settings",
  lead: "Preferences for this computer.",
  settingsLabel: "Settings",
  section: "Appearance",
  clusterTitle: "Machines",
  clusterDesc: "Adding and removing machines, and who may call this computer, are on the Machines page.",
  clusterLink: "Open the Machines page",
  storageTitle: "Storage",
  storageDesc: "What the app keeps on this computer's disk, and what you can delete to free space.",
  storageLink: "Open the Storage page",
  sectionLead: "Pick light or dark, and an accent colour. Kept in this browser.",
  modeTitle: "Mode",
  modeDesc: "System follows your computer's light or dark setting.",
  modes: { light: "Light", dark: "Dark", system: "System" } satisfies Record<Mode, string>,
  accentTitle: "Accent",
  accentDesc: "The colour of buttons, links, highlights and charts. It works in both modes.",
  accents: { neutral: "Neutral", ocean: "Ocean", ember: "Ember", forest: "Forest" } satisfies Record<Accent, string>,
  voiceTitle: "Reply voice",
  voiceDesc: "Choose the voice used when you play a spoken reply. Kept in this browser.",
};
