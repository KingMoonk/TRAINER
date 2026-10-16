"use client";

import { Button } from "@/components/ui/button";

/** Shared prompt and explicit checkpoint selection for model inspectors. */
export function InspectControls({ text, onText, version, onVersion, versions, pending, onSubmit, labels }: {
  text: string; onText: (text: string) => void;
  version: number | undefined; onVersion: (number: number) => void;
  versions: { number: number; name?: string | null }[];
  pending: boolean; onSubmit: () => void;
  labels: { prompt: string; version: string; inspect: string; running: string };
}) {
  return <form className="flex flex-col gap-3" onSubmit={(event) => { event.preventDefault(); onSubmit(); }}>
    <label className="flex flex-col gap-2 text-sm">{labels.version}
      <select value={version ?? ""} onChange={(event) => onVersion(Number(event.target.value))}
        disabled={!versions.length} className="h-10 max-w-full rounded-lg border border-input bg-background px-3">
        {version === undefined && <option value="" disabled>{versions.length ? "Choose a version" : "—"}</option>}
        {versions.map((item) => <option key={item.number} value={item.number}>Version {item.number}{item.name ? ` · ${item.name}` : ""}</option>)}
      </select>
    </label>
    <label className="flex flex-col gap-2 text-sm">{labels.prompt}
      <textarea value={text} onChange={(event) => onText(event.target.value)} rows={3}
        className="w-full min-w-0 resize-y rounded-lg border border-input bg-transparent p-3 text-base focus-visible:outline-ring" />
    </label>
    <Button type="submit" className="self-start" disabled={pending || !text.trim() || version === undefined}>
      {pending ? labels.running : labels.inspect}
    </Button>
  </form>;
}
