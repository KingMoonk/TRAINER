"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Pause, Play, SkipBack, SkipForward } from "lucide-react";
import dynamic from "next/dynamic";
import { useCallback, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { HEAD_STAGES, LEVELS, MODES, TILE_KINDS, isMatrixGroup, type HeadStage, type Level, type Mode, type TileKind } from "@/lib/scene";
import { fetchScene, runProbeStep, type SceneQuery } from "@/lib/viewer";

// WebGL does not exist while the static export prerenders, so the canvas only ever loads in the browser.
const SceneCanvas = dynamic(() => import("@/components/inside-3d/scene-canvas"), {
  ssr: false,
  loading: () => <div className="grid h-full place-items-center text-sm text-muted-foreground">Loading the 3D view...</div>,
});

const STAGE_HELP: Record<HeadStage, string> = {
  scores: "queries times keys",
  scale: "divided by the square root of the head size",
  mask: "later keys hidden",
  pattern: "softmax: where each query looks",
  values: "what the head passes on",
};

export function Inside3d() {
  const [level, setLevel] = useState<Level>("Model");
  const [mode, setMode] = useState<Mode>("Build");
  const [headStage, setHeadStage] = useState<HeadStage>("pattern");
  const [tileKind, setTileKind] = useState<TileKind>("weight");
  const [snapshot, setSnapshot] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [probing, setProbing] = useState(false);
  const [note, setNote] = useState("");
  const [reduceMotion, setReduceMotion] = useState(false);
  const [showNumbers, setShowNumbers] = useState(true);

  const query: SceneQuery = { level, mode, headStage, tileKind, snapshot };
  const { data: scene, error, isPending } = useQuery({
    queryKey: ["viewer-scene", query],
    queryFn: () => fetchScene(query),
    placeholderData: keepPreviousData,
  });

  const count = scene?.playback.count ?? 0;
  const last = Math.max(0, count - 1);

  const onPassDone = useCallback(() => {
    // Play moves on to the next snapshot and stops after the last one.
    if (snapshot < last) setSnapshot(snapshot + 1);
    else setPlaying(false);
  }, [snapshot, last]);

  const togglePlay = () => {
    if (!count) return setNote("Nothing to play yet: load snapshots first.");
    if (!playing && snapshot >= last) setSnapshot(0); // at the end, Play starts over
    setNote("");
    setPlaying(!playing);
  };

  const probe = async () => {
    if (probing) return;
    setProbing(true);
    setPlaying(false);
    setNote("Running one real training step on this version's checkpoint...");
    try {
      const result = await runProbeStep();
      setSnapshot(result.snapshot);
      setNote(result.note);
    } catch (reason) {
      setNote(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setProbing(false);
    }
  };

  const matrix = scene && Object.values(scene.groups).find(isMatrixGroup);
  const caption = matrix ? (showNumbers ? matrix.caption : matrix.caption.split("\n")[0]) : "";
  const rows = scene?.view.controls.rows ?? ["level", "mode"];
  const stageIndex = HEAD_STAGES.indexOf(headStage);

  return (
    <main className="flex h-dvh min-w-0 flex-col md:flex-row">
      <section className="relative min-h-80 flex-1 bg-[#0e1014]" aria-label="Inside the network">
        {scene ? (
          <SceneCanvas scene={scene} playing={playing} reduceMotion={reduceMotion} showNumbers={showNumbers} onPassDone={onPassDone} />
        ) : (
          <div className="grid h-full place-items-center text-sm text-muted-foreground">
            {error ? "Could not load the scene." : isPending ? "Loading the scene..." : ""}
          </div>
        )}
        {scene && (
          <div className="pointer-events-none absolute left-3 top-3 flex max-w-[calc(100%-1.5rem)] flex-col gap-1 font-mono text-xs text-foreground/90">
            <div>{scene.stats.info}</div>
            {scene.stats.training_line && <div className="text-muted-foreground">{scene.stats.training_line}</div>}
            {caption && <div className="whitespace-pre-line text-muted-foreground">{caption}</div>}
            {matrix?.weight_name && (
              <div className="text-muted-foreground">
                {matrix.weight_name}
                {matrix.shape ? ` (${matrix.shape.join(" x ")})` : ""}
              </div>
            )}
          </div>
        )}
        <p className="pointer-events-none absolute bottom-2 left-3 right-3 text-[11px] text-muted-foreground">
          Drag: look around | Wheel: zoom | Right-drag: move
        </p>
      </section>

      <aside className="flex w-full shrink-0 flex-col gap-3 overflow-y-auto border-t p-3 md:w-80 md:border-l md:border-t-0">
        <header>
          <p className="text-xs text-muted-foreground">Network designer › Look inside</p>
          <h1 className="text-lg font-semibold tracking-tight">Inside the network</h1>
        </header>

        {error && (
          <p role="alert" className="rounded-md border border-destructive/40 p-2 text-sm text-destructive">
            {error instanceof Error ? error.message : "The scene could not be loaded."}
          </p>
        )}

        <Card size="sm">
          <CardHeader>
            <CardTitle>View</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {rows.includes("level") && (
              <Field label="Zoom">
                <Choice values={LEVELS} value={level} onChange={setLevel} />
              </Field>
            )}
            {rows.includes("mode") && (
              <Field label="Mode">
                <Choice values={MODES} value={mode} onChange={setMode} />
              </Field>
            )}
            {rows.includes("stage") && (
              <Field label="Head step">
                <div className="flex items-center gap-1">
                  <Button
                    variant="outline"
                    size="icon"
                    aria-label="Previous head step"
                    disabled={stageIndex <= 0}
                    onClick={() => setHeadStage(HEAD_STAGES[stageIndex - 1])}
                  >
                    <ChevronLeft />
                  </Button>
                  <Select value={headStage} onValueChange={(value) => setHeadStage(value as HeadStage)}>
                    <SelectTrigger className="min-w-0 flex-1" aria-label="Head step">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {HEAD_STAGES.map((stage) => (
                        <SelectItem key={stage} value={stage}>
                          {stage}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    size="icon"
                    aria-label="Next head step"
                    disabled={stageIndex >= HEAD_STAGES.length - 1}
                    onClick={() => setHeadStage(HEAD_STAGES[stageIndex + 1])}
                  >
                    <ChevronRight />
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">{STAGE_HELP[headStage]}</p>
              </Field>
            )}
            {rows.includes("tile") && (
              <Field label="Show">
                <Choice values={TILE_KINDS} value={tileKind} onChange={setTileKind} />
              </Field>
            )}
          </CardContent>
        </Card>

        <Card size="sm">
          <CardHeader>
            <CardTitle>Replay</CardTitle>
            {scene?.playback.show && (
              <Badge variant="secondary" className="ml-auto">
                {scene.playback.position}
              </Badge>
            )}
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {scene?.playback.show ? (
              <div className="grid grid-cols-3 gap-2">
                <Button variant="outline" disabled={snapshot <= 0 || probing} onClick={() => (setPlaying(false), setSnapshot(snapshot - 1))}>
                  <SkipBack /> Previous
                </Button>
                <Button disabled={probing} onClick={togglePlay}>
                  {playing ? <Pause /> : <Play />} {playing ? "Pause" : "Play"}
                </Button>
                <Button variant="outline" disabled={snapshot >= last || probing} onClick={() => (setPlaying(false), setSnapshot(snapshot + 1))}>
                  Next <SkipForward />
                </Button>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">{scene?.playback.position ?? "No snapshots loaded."}</p>
            )}
            <Button variant="secondary" disabled={probing || scene?.view.controls.snapshots.show === false} onClick={probe}>
              {probing ? "Running one step..." : "Run one real step"}
            </Button>
            {(note || scene?.stats.note) && (
              <p role="status" className="text-xs text-muted-foreground">
                {note || scene?.stats.note}
              </p>
            )}
            <div className="flex flex-col gap-2">
              {scene?.view.controls.toggles.includes("motion") && (
                <div className="flex items-center justify-between">
                  <Label htmlFor="motion">Motion</Label>
                  <Switch id="motion" checked={!reduceMotion} onCheckedChange={(on) => setReduceMotion(!on)} />
                </div>
              )}
              {scene?.view.controls.toggles.includes("numbers") && (
                <div className="flex items-center justify-between">
                  <Label htmlFor="numbers">Numbers</Label>
                  <Switch id="numbers" checked={showNumbers} onCheckedChange={setShowNumbers} />
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <p className="text-xs text-muted-foreground">
          Showing a recorded scene from <code>src/fixtures/scene.json</code> until the live viewer API lands.
        </p>
      </aside>
    </main>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{label}</span>
      {children}
    </div>
  );
}

/** A single-choice toggle group; clicking the chosen item again keeps it chosen. */
function Choice<T extends string>({ values, value, onChange }: { values: readonly T[]; value: T; onChange: (value: T) => void }) {
  return (
    <ToggleGroup
      type="single"
      variant="outline"
      size="sm"
      value={value}
      onValueChange={(next) => next && onChange(next as T)}
      className="flex w-full flex-wrap"
    >
      {values.map((item) => (
        <ToggleGroupItem key={item} value={item} className="flex-1 first-letter:uppercase">
          {item}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}
