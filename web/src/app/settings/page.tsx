"use client";

import { Check } from "lucide-react";
import { useTheme } from "next-themes";
import { useSyncExternalStore } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useAccent } from "@/components/theme";
import { APPEARANCE } from "@/lib/copy/appearance";
import { MachinesSection } from "./machines";
import { ACCENTS, MODES, type Accent, type Mode } from "@/lib/theme";

// The chosen button: an accent border and ring on a raised fill, plus a tick. Tokens only, so it reads in every mode
// and accent (a filled `bg-primary` would vanish for Neutral in dark, where primary is near-white like the swatch).
const SELECTED = "data-[state=on]:border-primary data-[state=on]:bg-accent data-[state=on]:ring-1 data-[state=on]:ring-primary data-[state=on]:hover:bg-accent";
const CHECK = <Check aria-hidden="true" className="hidden size-3.5 group-data-[state=on]/toggle:inline" />;

const mounted = () => () => {};

export default function Page() {
  const { theme, resolvedTheme, setTheme } = useTheme();
  const [accent, setAccent] = useAccent();
  // next-themes only knows the stored mode on the client; show nothing selected until then rather than the wrong one.
  const ready = useSyncExternalStore(mounted, () => true, () => false);
  const dark = resolvedTheme !== "light";
  return (
    <main className="mx-auto flex w-full max-w-5xl flex-col gap-6 p-8">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold tracking-tight">{APPEARANCE.h1}</h1>
        <p className="max-w-prose text-muted-foreground">{APPEARANCE.lead}</p>
      </div>
      <section aria-labelledby="appearance" className="flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          <h2 id="appearance" className="text-lg font-semibold tracking-tight">{APPEARANCE.section}</h2>
          <p className="max-w-prose text-sm text-muted-foreground">{APPEARANCE.sectionLead}</p>
        </div>
      <Card>
        <CardHeader>
          <CardTitle>{APPEARANCE.modeTitle}</CardTitle>
          <CardDescription>{APPEARANCE.modeDesc}</CardDescription>
        </CardHeader>
        <CardContent>
          <ToggleGroup type="single" variant="outline" aria-label={APPEARANCE.modeTitle} data-testid="appearance-mode"
            value={ready ? theme : ""} onValueChange={(value) => value && setTheme(value as Mode)}>
            {MODES.map((mode) => <ToggleGroupItem key={mode} value={mode} className={SELECTED}>{CHECK}{APPEARANCE.modes[mode]}</ToggleGroupItem>)}
          </ToggleGroup>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>{APPEARANCE.accentTitle}</CardTitle>
          <CardDescription>{APPEARANCE.accentDesc}</CardDescription>
        </CardHeader>
        <CardContent>
          <ToggleGroup type="single" variant="outline" aria-label={APPEARANCE.accentTitle} data-testid="appearance-accent"
            value={accent} onValueChange={(value) => value && setAccent(value as Accent)}>
            {ACCENTS.map((id) => (
              <ToggleGroupItem key={id} value={id} className={SELECTED}>
                {CHECK}
                {/* A swatch carries its own data-theme (and .dark) so it shows that accent's primary, whatever is active. */}
                <span aria-hidden="true" data-theme={id === "neutral" ? undefined : id} className={`size-3 rounded-full border border-foreground/30 bg-primary ${dark ? "dark" : ""}`} />
                {APPEARANCE.accents[id]}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </CardContent>
      </Card>
      </section>
      <MachinesSection />
    </main>
  );
}
