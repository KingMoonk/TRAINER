"use client";

import { useTheme } from "next-themes";
import { useSyncExternalStore } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useAccent } from "@/components/theme";
import { APPEARANCE } from "@/lib/copy/appearance";
import { ACCENTS, MODES, type Accent, type Mode } from "@/lib/theme";

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
      <Card>
        <CardHeader>
          <CardTitle>{APPEARANCE.modeTitle}</CardTitle>
          <CardDescription>{APPEARANCE.modeDesc}</CardDescription>
        </CardHeader>
        <CardContent>
          <ToggleGroup type="single" variant="outline" aria-label={APPEARANCE.modeTitle} data-testid="appearance-mode"
            value={ready ? theme : ""} onValueChange={(value) => value && setTheme(value as Mode)}>
            {MODES.map((mode) => <ToggleGroupItem key={mode} value={mode}>{APPEARANCE.modes[mode]}</ToggleGroupItem>)}
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
              <ToggleGroupItem key={id} value={id}>
                {/* A swatch carries its own data-theme (and .dark) so it shows that accent's primary, whatever is active. */}
                <span aria-hidden="true" data-theme={id === "neutral" ? undefined : id} className={`size-3 rounded-full bg-primary ${dark ? "dark" : ""}`} />
                {APPEARANCE.accents[id]}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </CardContent>
      </Card>
    </main>
  );
}
