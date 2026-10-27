"use client";

import { Check } from "lucide-react";
import { useTheme } from "next-themes";
import { useSyncExternalStore } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useAccent } from "@/components/theme";
import { APPEARANCE } from "@/lib/copy/appearance";
import Link from "next/link";
import { routeFor } from "@/lib/routes";
import { ACCENTS, MODES, type Accent, type Mode } from "@/lib/theme";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";
import { TTS_VOICES, useTtsVoice, type TtsVoice } from "@/lib/voice";

// The chosen button: the app-wide neutral selected look (globals.css), plus a tick.
const SELECTED = "data-[state=on]:selected-box data-[state=on]:hover:bg-(--selected-surface)";
const CHECK = <Check aria-hidden="true" className="hidden size-3.5 group-data-[state=on]/toggle:inline" />;

const mounted = () => () => {};

export default function Page() {
  const { theme, resolvedTheme, setTheme } = useTheme();
  const [accent, setAccent] = useAccent();
  const [ttsVoice, setTtsVoice] = useTtsVoice();
  // next-themes only knows the stored mode on the client; show nothing selected until then rather than the wrong one.
  const ready = useSyncExternalStore(mounted, () => true, () => false);
  const dark = resolvedTheme !== "light";
  return (
    <main className={PAGE_MAIN}>
      <PageHeader title={APPEARANCE.h1} lead={APPEARANCE.lead} />
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
      <Card data-testid="settings-voice">
        <CardHeader>
          <CardTitle>{APPEARANCE.voiceTitle}</CardTitle>
          <CardDescription>{APPEARANCE.voiceDesc}</CardDescription>
        </CardHeader>
        <CardContent>
          <Select value={ttsVoice} onValueChange={(value) => setTtsVoice(value as TtsVoice)}>
            <SelectTrigger aria-label={APPEARANCE.voiceTitle} className="w-full max-w-md">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {TTS_VOICES.map(({ id, label }) => <SelectItem key={id} value={id}>{label}</SelectItem>)}
            </SelectContent>
          </Select>
        </CardContent>
      </Card>
      <Card data-testid="settings-cluster">
        <CardHeader>
          <CardTitle>{APPEARANCE.clusterTitle}</CardTitle>
          <CardDescription>{APPEARANCE.clusterDesc}</CardDescription>
        </CardHeader>
        <CardContent>
          <Link href={routeFor("machines").href} className="text-sm underline underline-offset-2">{APPEARANCE.clusterLink}</Link>
        </CardContent>
      </Card>
      <Card data-testid="settings-storage">
        <CardHeader>
          <CardTitle>{APPEARANCE.storageTitle}</CardTitle>
          <CardDescription>{APPEARANCE.storageDesc}</CardDescription>
        </CardHeader>
        <CardContent>
          <Link href={routeFor("storage").href} className="text-sm underline underline-offset-2">{APPEARANCE.storageLink}</Link>
        </CardContent>
      </Card>
    </main>
  );
}
