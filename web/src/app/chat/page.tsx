"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { AvailableVersions } from "@/components/available-versions";
import { api, ApiError, subscribe, unwrap } from "@/lib/api";
import type { components } from "@/lib/api-schema";
import { availableElsewhere, cameFrom, isCluster, useCluster, usePublished } from "@/lib/cluster";
import { routeFor } from "@/lib/routes";
import { postStream } from "@/lib/sse-post";
import { CHAT as T } from "./copy";
import { CheckpointStep, FewShotCheck, checkKey, useBestVersion } from "./checks";
import {
  canRate,
  changesCheckpoints,
  compareStart,
  scoreRows,
  trainedVersions,
  unfinished,
  verdictView,
  versionName,
  type Message,
  type Version,
  type VerdictResult,
} from "./logic";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";

type PaneState = { text: string; status: "idle" | "streaming" | "done" | "stopped" | "error"; error?: string };
const IDLE: PaneState = { text: "", status: "idle" };

/** A sentence for a failed call: the server's own message for a 4xx, else what the browser said. */
function describe(error: unknown): string {
  if (error instanceof ApiError) return `The server answered ${error.status}${error.body ? `: ${error.body}` : ""}`;
  return error instanceof Error ? error.message : "Something went wrong.";
}

export default function Page() {
  const route = routeFor("chat");
  const queryClient = useQueryClient();
  const versionsQuery = useQuery({
    queryKey: ["versions"],
    queryFn: () => unwrap(api.GET("/versions")),
    refetchInterval: 15_000, // a version finishing training appears on its own
  });
  const cluster = useCluster().data;
  const published = usePublished(cluster).data;
  const multi = isCluster(cluster); // a cluster of one shows nothing about machines
  const trained = trainedVersions(versionsQuery.data ?? []);
  const numbers = trained.map((v) => v.number);
  const byNumber = new Map(trained.map((v) => [v.number, v]));
  const best = useBestVersion(trained);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const close = subscribe((event) => {
      if (!changesCheckpoints(event)) return;
      clearTimeout(timer);
      timer = setTimeout(() => {
        void queryClient.invalidateQueries({ queryKey: ["versions"] });
        void queryClient.invalidateQueries({ queryKey: ["chat", "best"] });
        void queryClient.invalidateQueries({ queryKey: ["chat", "checkpoint"] });
        // A checkpoint can be rewritten at the same path; never reuse its old example-check result.
        void queryClient.resetQueries({ queryKey: ["chat", "few-shot"] });
      }, 250);
    });
    return () => { close(); clearTimeout(timer); };
  }, [queryClient]);

  // null = follow the actual best resolved by the server's scoring job.
  const [picked, setPicked] = useState<number | null>(null);
  const [compare, setCompare] = useState<[number, number] | null>(null);
  const [thread, setThread] = useState<Message[]>([]);
  const [panes, setPanes] = useState<[PaneState, PaneState]>([IDLE, IDLE]);
  const [draft, setDraft] = useState("");
  const [teachOpen, setTeachOpen] = useState(false);
  const [correction, setCorrection] = useState("");
  const [whole, setWhole] = useState(false);
  const [how, setHow] = useState(false);
  const [replyVersion, setReplyVersion] = useState<number | null>(null);
  const controllers = useRef<(AbortController | null)[]>([null, null]);

  useEffect(() => {
    const live = controllers.current;
    return () => live.forEach((c, i) => { c?.abort(); live[i] = null; });
  }, []);

  // A picked version that stopped being trained (or vanished) goes back to following the best one.
  const following = picked === null || !byNumber.has(picked);
  const resolved = following ? best.data ?? null : picked;
  const busy = panes.some((p) => p.status === "streaming");
  const answering = busy && !compare ? replyVersion : resolved;
  const left = compare ? byNumber.get(compare[0]) : undefined;
  const right = compare ? byNumber.get(compare[1]) : undefined;
  const ready = compare ? !!left && !!right : answering !== null && byNumber.has(answering) && (!following || (!best.isFetching && !best.isError));

  const setPane = (i: number, next: PaneState) => setPanes((p) => (i === 0 ? [next, p[1]] : [p[0], next]));

  async function run(i: number, version: number, messages: Message[], single: boolean) {
    const ctl = new AbortController();
    controllers.current[i] = ctl;
    let text = "";
    let finished = false;
    let failed: string | null = null;
    setPane(i, { text: "", status: "streaming" });
    try {
      await postStream(
        "/chat/reply",
        { version, messages } satisfies components["schemas"]["ReplyRequest"],
        (event) => {
          if (controllers.current[i] !== ctl) return;
          const payload = (event.payload ?? {}) as { text?: string; message?: string };
          if (event.kind === "text" || event.kind === "done") {
            text = payload.text ?? text;
            if (event.kind === "done") finished = true;
            else setPane(i, { text, status: "streaming" });
          } else if (event.kind === "error") {
            failed = payload.message ?? "The reply failed.";
          }
        },
        ctl.signal,
      );
    } catch (error) {
      failed = describe(error);
    }
    if (controllers.current[i] !== ctl) return;
    controllers.current[i] = null;
    if (!failed && !finished && !ctl.signal.aborted) failed = "The reply ended early.";
    if (failed) {
      toast.error(failed);
      setPane(i, { text, status: "error", error: failed });
      return;
    }
    // Single mode keeps the conversation: the finished (or stopped) reply joins the thread and the live bubble clears.
    if (single) {
      if (text) setThread((t) => [...t, { role: "assistant", content: text }]);
      setPane(i, IDLE);
    } else {
      setPane(i, { text, status: ctl.signal.aborted ? "stopped" : "done" });
    }
  }

  function send() {
    const content = draft.trim();
    if (!content || busy || !ready) return;
    const message: Message = { role: "user", content };
    const messages = compare ? [message] : [...thread, message];
    setDraft("");
    setThread(messages);
    setPanes([IDLE, IDLE]);
    setReplyVersion(compare ? compare[0] : answering);
    if (compare) {
      void run(0, compare[0], messages, false);
      void run(1, compare[1], messages, false);
    } else {
      void run(0, answering!, messages, true);
    }
  }

  function stop() {
    controllers.current.forEach((c) => c?.abort());
  }

  function reset(nextCompare: [number, number] | null) {
    stop();
    controllers.current[0] = controllers.current[1] = null;
    setCompare(nextCompare);
    setThread([]);
    setPanes([IDLE, IDLE]);
    setReplyVersion(null);
  }

  function pick(number: number) {
    if (busy) return;
    if (compare) {
      if (number !== compare[0] && number !== compare[1]) {
        stop();
        controllers.current[0] = controllers.current[1] = null;
        setCompare([compare[0], number]);
        setPanes([IDLE, IDLE]);
        setThread([]);
        setReplyVersion(null);
      }
    } else {
      setPicked(number);
    }
  }

  const replies: [string, string] = [panes[0].text, panes[1].text];
  const rateEnabled = !!compare && panes.every((pane) => pane.status === "done") && canRate(replies, busy);

  const rate = useMutation({
    mutationFn: async (chosenIndex: 0 | 1) => {
      if (!compare) return chosenIndex;
      const other = chosenIndex === 0 ? 1 : 0;
      await unwrap(
        api.POST("/chat/pairs", {
          body: {
            messages: thread,
            chosen: { version: compare[chosenIndex], reply: replies[chosenIndex] },
            rejected: { version: compare[other], reply: replies[other] },
          },
        }),
      );
      return chosenIndex;
    },
    onSuccess: (chosenIndex) => {
      if (compare) toast.success(`Saved: Version ${compare[chosenIndex]} was better. Used in Post-training round 4.`);
    },
    onError: (error) => toast.error(describe(error)),
  });

  // A correction belongs to a reply: the context is the conversation up to the last thing the person said.
  const lastUser = thread.map((m) => m.role).lastIndexOf("user");
  const teach = useMutation({
    mutationFn: () =>
      unwrap(
        api.POST("/chat/corrections", {
          body: {
            version: replyVersion,
            messages: thread.slice(0, lastUser + 1),
            correction: correction.trim(),
            whole_answer: whole,
          },
        }),
      ),
    onSuccess: () => {
      toast.success(whole ? "Saved for Post-training round 1 as a whole new answer." : "Saved for Post-training round 3.");
      setCorrection("");
    },
    onError: (error) => toast.error(describe(error)),
  });

  const verdictReady = !!left && !!right && !unfinished(left) && !unfinished(right);
  const verdictQuery = useQuery({
    queryKey: ["verdict", compare?.[0], compare?.[1]],
    enabled: verdictReady,
    queryFn: () =>
      unwrap(api.GET("/benchmark/verdict", { params: { query: { a: Math.min(...compare!), b: Math.max(...compare!) } } })),
  });

  const scored = compare ? left : answering !== null ? byNumber.get(answering) : undefined;

  return (
    <main className={PAGE_MAIN}>
      <PageHeader title={route.title} lead={route.lead} />

      {versionsQuery.isPending ? (
        <div className="flex flex-col gap-3" aria-label="Loading versions">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-40 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      ) : versionsQuery.isError && !versionsQuery.data ? (
        <Card>
          <CardContent className="flex flex-col items-start gap-3">
            <p className="text-sm">Could not load the versions: {describe(versionsQuery.error)}</p>
            <Button variant="outline" size="sm" onClick={() => void versionsQuery.refetch()}>
              Try again
            </Button>
          </CardContent>
        </Card>
      ) : (
        <>
          {versionsQuery.isError && <p role="alert">Could not refresh the versions: {describe(versionsQuery.error)}</p>}
          <Card>
            <CardContent className="flex flex-wrap items-center gap-3">
              <Badge variant="outline">Live</Badge>
              <span className="text-lg font-semibold">
                {numbers.length ? (compare ? "Two versions" : answering !== null ? `Version ${answering}` : best.isFetching ? T.findingBest : "Pick a trained version") : T.noTrained}
              </span>
              {numbers.length > 0 && !compare && (
                <span className="text-sm text-muted-foreground">{following ? T.follows : T.picked}</span>
              )}
              {best.data != null && !best.isFetching && !best.isError && <span className="text-sm text-muted-foreground">{T.best(best.data)}</span>}
              {!numbers.length && (
                <Link href="/builder" className="text-sm text-primary underline-offset-4 hover:underline">
                  {T.noTrainedLink}
                </Link>
              )}
              <div className="ml-auto flex gap-2">
                {!compare && !following && (
                  <Button variant="ghost" size="sm" disabled={busy} onClick={() => setPicked(null)}>
                    Follow the best checkpoint
                  </Button>
                )}
                <Button
                  variant="outline"
                  size="sm"
                  disabled={busy || (!compare && (numbers.length < 2 || !ready))}
                  onClick={() => reset(compare ? null : compareStart(numbers, answering))}
                >
                  {compare ? T.backToOne : T.compareBtn}
                </Button>
              </div>
            </CardContent>
          </Card>

          {numbers.length > 0 && <div className="flex flex-wrap items-center gap-2 text-sm">
            {best.isFetching && <span role="status" className="flex items-center gap-2"><Skeleton className="h-4 w-20" />{T.findingBest}</span>}
            {best.isError && <p role="alert">{T.bestFailed}: {describe(best.error)}</p>}
            {best.data === null && !best.isFetching && !best.isError && <p>{T.noBest}</p>}
            <Button variant="ghost" size="sm" disabled={best.isFetching || busy} onClick={() => void best.refetch()}>{T.retryBest}</Button>
          </div>}

          <section aria-label="Replies" className={compare ? "grid gap-4 md:grid-cols-2" : "flex flex-col gap-4"}>
            {compare ? (
              ([0, 1] as const).map((i) => {
                const version = i === 0 ? left : right;
                return (
                  <Card key={i}>
                    <CardHeader>
                      <CardTitle className={i === 0 ? "text-side-left" : "text-side-right"}>
                        {version ? versionName(version) : "No version"}
                      </CardTitle>
                      {version && <CheckpointStep version={version} />}
                    </CardHeader>
                    <CardContent className="flex flex-col gap-3">
                      <ReplyText pane={panes[i]} />
                      <Button
                        variant="outline"
                        size="sm"
                        className="self-start"
                        disabled={!rateEnabled || rate.isPending}
                        title={T.pickingReplyDesc}
                        onClick={() => rate.mutate(i)}
                      >
                        {T.thisOneBetter}
                      </Button>
                    </CardContent>
                  </Card>
                );
              })
            ) : (
              <Card>
                {scored && <CardHeader><CardTitle>{versionName(scored)}</CardTitle><CheckpointStep version={scored} /></CardHeader>}
                <CardContent className="flex min-h-40 flex-col gap-3">
                  {thread.length === 0 && panes[0].status === "idle" && <p className="text-sm text-muted-foreground">{T.itsReply}</p>}
                  {thread.map((m, k) => (
                    <Bubble key={k} role={m.role} text={m.content} />
                  ))}
                  {panes[0].status !== "idle" && <Bubble role="assistant" text={panes[0].text} pane={panes[0]} />}
                </CardContent>
              </Card>
            )}
            {compare && thread[0] && (
              <p className="text-sm text-muted-foreground md:col-span-2">
                <span className="font-medium text-foreground">You:</span> {thread[0].content}
              </p>
            )}
          </section>

          <form
            className="flex gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              send();
            }}
          >
            <input
              aria-label="Your message"
              className="h-9 min-w-0 flex-1 rounded-lg border bg-transparent px-3 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50"
              placeholder={compare ? T.placeholderCompare : T.placeholderSingle}
              value={draft}
              disabled={!ready}
              onChange={(e) => setDraft(e.target.value)}
            />
            {busy ? (
              <Button type="button" variant="destructive" onClick={stop}>
                {T.stop}
              </Button>
            ) : (
              <Button type="submit" disabled={!ready || !draft.trim()}>
                {T.send}
              </Button>
            )}
          </form>

          <section className="flex flex-col gap-2">
            <button
              type="button"
              className="self-start text-sm text-primary underline-offset-4 hover:underline"
              aria-expanded={teachOpen}
              onClick={() => setTeachOpen((o) => !o)}
            >
              {T.teachCorrection}
            </button>
            {teachOpen && (
              <form
                className="flex flex-col gap-3 rounded-lg border p-3"
                onSubmit={(e) => {
                  e.preventDefault();
                  if (correction.trim() && lastUser >= 0 && replyVersion !== null && !busy) teach.mutate();
                }}
              >
                <p className="text-sm text-muted-foreground">{T.teachCorrectionDesc}</p>
                <input
                  aria-label="Correction"
                  className="h-9 rounded-lg border bg-transparent px-3 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
                  placeholder="What it should have said"
                  value={correction}
                  onChange={(e) => setCorrection(e.target.value)}
                />
                <div className="flex flex-wrap items-center gap-4">
                  <div className="flex items-center gap-2">
                    <Switch id="whole-answer" checked={whole} onCheckedChange={setWhole} />
                    <Label htmlFor="whole-answer">{T.wholeAnswer}</Label>
                  </div>
                  <Button type="submit" size="sm" disabled={!correction.trim() || lastUser < 0 || replyVersion === null || busy || teach.isPending}>
                    Save
                  </Button>
                  {lastUser < 0 && <span className="text-sm text-muted-foreground">Send a message first; a correction belongs to a reply.</span>}
                </div>
              </form>
            )}
          </section>

          {compare && left && right && (
            <Verdict a={left} b={right} how={how} onHow={() => setHow((h) => !h)} ready={verdictReady} query={verdictQuery} />
          )}

          {trained.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">{T.versions}</CardTitle>
                <p className="text-sm text-muted-foreground">{compare ? T.shelfHintCompare : T.shelfHintSingle}</p>
              </CardHeader>
              <CardContent className="flex flex-col gap-2">
                {trained.map((v) => {
                  const isLeft = compare ? compare[0] === v.number : answering === v.number;
                  const isRight = !!compare && compare[1] === v.number;
                  const slot = isLeft
                    ? compare
                      ? "left reply"
                      : "answering now"
                    : isRight
                      ? "right reply"
                      : v.status === "training"
                        ? "still training · answers from its best checkpoint so far"
                        : "";
                  const chip = v.status === "training" ? "catching up" : v.verdict || (v.parent == null ? "baseline" : "not compared yet");
                  return (
                    <div key={v.number} className="rounded-lg border p-3">
                    <button
                      type="button"
                      disabled={busy}
                      aria-pressed={isLeft || isRight}
                      onClick={() => pick(v.number)}
                      className={`flex w-full flex-wrap items-center gap-3 rounded-lg border p-3 text-left text-sm transition-colors hover:bg-muted disabled:opacity-50 ${
                        isLeft || isRight ? "selected-box" : ""
                      }`}
                    >
                      <span className="w-24 font-semibold">{versionName(v)}</span>
                      <span className="w-32 text-muted-foreground">{chip}</span>
                      <span className="flex-1">{v.change}</span>
                      {multi && (
                        <span className="text-xs text-muted-foreground" data-testid={`where-${v.number}`}>
                          on this machine{cameFrom(published, v.number) ? ` · pulled from ${cameFrom(published, v.number)}` : ""}
                        </span>
                      )}
                      <span className={isLeft ? "text-side-left" : isRight ? "text-side-right" : "text-muted-foreground"}>{slot}</span>
                    </button>
                    <div className="mt-2"><CheckpointStep version={v} /></div>
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          )}

          {multi && availableElsewhere(published).length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">Available on other machines</CardTitle>
                <p className="text-sm text-muted-foreground">Pull one to chat with it here.</p>
              </CardHeader>
              <CardContent>
                <AvailableVersions published={published} />
              </CardContent>
            </Card>
          )}

          {scored && (
            <Card>
              <CardContent className="flex flex-col gap-1">
                {scoreRows(scored).map(([label, value]) => (
                  <div key={label} className="flex gap-4 text-sm">
                    <span className="w-72 text-muted-foreground">{label}</span>
                    <span className="font-mono">{value}</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
          {scored && <FewShotCheck key={JSON.stringify(checkKey("few-shot", scored))} version={scored} />}

          <div className="flex flex-wrap gap-2">
            <Link href="/compare" className={buttonVariants()}>
              {T.testChange}
            </Link>
            <Link href="/builder" className={buttonVariants({ variant: "outline" })}>
              {T.openDesigner}
            </Link>
            <Link href="/benchmark" className={buttonVariants({ variant: "outline" })}>
              Open the benchmark
            </Link>
          </div>
        </>
      )}
    </main>
  );
}

function ReplyText({ pane }: { pane: PaneState }) {
  if (pane.status === "idle") return <p className="text-sm text-muted-foreground">{T.itsReply}</p>;
  return (
    <div className="flex flex-col gap-2">
      {pane.text ? (
        <p className="whitespace-pre-wrap text-sm" aria-live="polite">
          {pane.text}
        </p>
      ) : (
        pane.status === "streaming" && <Skeleton className="h-4 w-2/3" />
      )}
      {pane.status === "stopped" && <span className="text-xs text-muted-foreground">Stopped</span>}
      {pane.status === "error" && <span className="text-xs text-destructive">{pane.error}</span>}
    </div>
  );
}

function Bubble({ role, text, pane }: { role: Message["role"]; text: string; pane?: PaneState }) {
  const mine = role === "user";
  return (
    <div className={`flex flex-col gap-1 ${mine ? "items-end" : "items-start"}`}>
      <div className={`max-w-[85%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm ${mine ? "bg-primary text-primary-foreground" : "bg-muted"}`}>
        {pane ? <ReplyText pane={pane} /> : text}
      </div>
    </div>
  );
}

function Verdict({
  a,
  b,
  how,
  onHow,
  query,
  ready,
}: {
  a: Version;
  b: Version;
  how: boolean;
  onHow: () => void;
  query: { isPending: boolean; isError: boolean; data?: VerdictResult };
  ready: boolean;
}) {
  const waiting = ready && query.isPending;
  const view = verdictView(a, b, query.isError ? { comparable: false, reason: "the server could not work it out." } : query.data);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">{T.verdict}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {waiting ? (
          <Skeleton className="h-6 w-2/3" />
        ) : (
          <>
            <p className={`text-lg font-semibold ${view.ok ? "" : "text-warning"}`}>{view.line}</p>
            <p className="text-sm text-muted-foreground">{view.sub}</p>
          </>
        )}
        <button type="button" className="self-start text-sm text-primary underline-offset-4 hover:underline" aria-expanded={how} onClick={onHow}>
          {how ? T.hideHowMeasured : T.howMeasured}
        </button>
        {how && (
          <div className="flex flex-col gap-2 text-xs text-muted-foreground">
            <p>{T.verdictText}</p>
            <p>{T.verdictErrorUnit}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
