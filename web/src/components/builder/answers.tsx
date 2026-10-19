"use client";

// "When answering" (builder.py _charts, ask_both): the same question put to the current version and the next one, each
// answering with its best checkpoint, the replies streaming in word by word from POST /chat/reply.

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { answerRows, detailOf, PROMPT, versionName, type Version } from "@/lib/builder";
import { BUILDER, fill } from "@/lib/copy/builder";
import { postStream } from "@/lib/sse-post";
import { cn } from "@/lib/utils";

type Reply = { text: string; state: "streaming" | "done" | "error" };

export function Answers({ versions, current }: { versions: Version[] | undefined; current: number | null }) {
  const [replies, setReplies] = useState<Record<number, Reply>>({});
  const controllers = useRef<AbortController[]>([]);
  const stopAll = () => {
    controllers.current.forEach((c) => c.abort());
    controllers.current = [];
  };
  useEffect(() => stopAll, []);

  if (!versions) return <Skeleton className="h-24 w-full" data-testid="answers-loading" />;
  const n = current ?? 0;
  const trained = new Set(versions.filter((v) => v.best_checkpoint).map((v) => v.number));
  const rows = answerRows(n, new Set([n, n + 1].filter((k) => trained.has(k))), !versions.some((v) => v.number === n + 1));
  const askable = rows.filter((row) => !row.why).map((row) => row.number);
  const running = Object.values(replies).some((r) => r.state === "streaming");

  const ask = () => {
    stopAll();
    setReplies(Object.fromEntries(askable.map((k) => [k, { text: "", state: "streaming" } as Reply])));
    for (const number of askable) {
      const ctl = new AbortController();
      controllers.current.push(ctl);
      const set = (reply: Reply) => setReplies((all) => (ctl.signal.aborted ? all : { ...all, [number]: reply }));
      let text = "";
      postStream(
        "/chat/reply",
        { version: number, messages: [{ role: "user", content: PROMPT }] },
        (event) => {
          const payload = (event.payload ?? {}) as { text?: string; message?: string };
          if (event.kind === "text" || event.kind === "done") {
            text = payload.text ?? text;
            set({ text, state: event.kind === "done" ? "done" : "streaming" });
          } else if (event.kind === "error") {
            set({ text: payload.message ?? BUILDER.answer_failed, state: "error" });
          }
        },
        ctl.signal,
      ).then(
        () => setReplies((all) => (ctl.signal.aborted || all[number]?.state !== "streaming" ? all : { ...all, [number]: { text, state: "done" } })),
        (reason) => {
          if (ctl.signal.aborted) return;
          toast.error(detailOf(reason));
          set({ text: detailOf(reason), state: "error" });
        },
      );
    }
  };
  const stop = () => {
    stopAll();
    setReplies((all) => Object.fromEntries(Object.entries(all).map(([k, r]) => [k, r.state === "streaming" ? { ...r, state: "done" } : r])));
  };

  return (
    <Card size="sm" data-testid="answers">
      <CardHeader>
        <CardTitle title={BUILDER.answering_note}>{BUILDER.when_answering}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {!rows.length ? (
          <p className="text-sm text-muted-foreground">{n ? fill(BUILDER.nothing_to_ask, { n }) : BUILDER.nothing_to_ask_first}</p>
        ) : (
          <>
            <p className="font-serif text-sm">{PROMPT} …</p>
            <ul className="flex flex-col gap-1">
              {rows.map((row) => {
                const reply = replies[row.number];
                return (
                  <li key={row.number} className="flex gap-3 text-sm" data-testid={`answer-${row.number}`}>
                    <span className={cn("w-24 shrink-0 font-medium", row.number === n ? "text-part-own" : "text-warning")}>{versionName(row.number)}</span>
                    {reply ? (
                      <span className={cn("font-serif", reply.state === "error" && "text-destructive")} aria-live="polite">
                        {reply.text || "…"}
                      </span>
                    ) : (
                      <span className="text-muted-foreground">{row.why || BUILDER.its_answer_appears}</span>
                    )}
                  </li>
                );
              })}
            </ul>
            <div>
              {running ? (
                <Button size="sm" variant="outline" onClick={stop}>
                  {BUILDER.answer_stop}
                </Button>
              ) : (
                <Button size="sm" variant="outline" onClick={ask} data-testid="ask-both">
                  {askable.length > 1 ? BUILDER.ask_both : fill(BUILDER.ask_one, { version: versionName(askable[0]) })}
                </Button>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
