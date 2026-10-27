"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api, unwrap } from "@/lib/api";
import type { components } from "@/lib/api-schema";
import { postStream } from "@/lib/sse-post";
import { errorText } from "./logic";

type Message = components["schemas"]["Message"];

/** A round's warning above the round: it would start from the pretrained checkpoint, not from a Round 1 result. */
export function BaseWarning({ text }: { text: string }) {
  return (
    <p role="alert" data-testid="base-warning" className="rounded-md border border-destructive/40 p-3 text-sm text-destructive">
      {text}
    </p>
  );
}

/**
 * A round that failed its gate keeps its checkpoint without making it live. Here it can be tried in a short chat (Chat itself keeps
 * following the live checkpoint) and, if it is good enough, adopted: it becomes the version's best checkpoint.
 */
export function KeptCard({
  round,
  version,
  onAdopted,
  adoptWith,
}: {
  round: number;
  version: number | undefined;
  onAdopted: () => void;
  /** Adopt some other way (the Auto Run page adopts through its run, which then goes on); it says what to toast. */
  adoptWith?: () => Promise<string>;
}) {
  const queryClient = useQueryClient();
  const [draft, setDraft] = useState("");
  const [thread, setThread] = useState<Message[]>([]);
  const [reply, setReply] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const control = useRef<AbortController | null>(null);
  useEffect(() => () => control.current?.abort(), []);

  const adopt = useMutation({
    mutationFn: async () => {
      if (adoptWith) return await adoptWith();
      const adopted = await unwrap(api.POST("/post-training/rounds/{round}/adopt", { params: { path: { round } }, body: { version: version ?? null } }));
      return `Version ${adopted.version} now uses Round ${adopted.round}'s checkpoint. Chat follows it from the next message.`;
    },
    onSuccess: (message) => {
      toast.success(message);
      queryClient.invalidateQueries({ queryKey: ["versions"] });
      onAdopted();
    },
    onError: (e) => toast.error(errorText(e)),
  });

  async function send() {
    const content = draft.trim();
    if (!content || reply !== null) return;
    const messages: Message[] = [...thread, { role: "user", content }];
    setThread(messages);
    setDraft("");
    setError(null);
    setReply("");
    const ctl = new AbortController();
    control.current = ctl;
    let text = "";
    let failed: string | null = null;
    try {
      await postStream(
        "/chat/reply",
        { version: version ?? null, messages, round } satisfies components["schemas"]["ReplyRequest"],
        (event) => {
          const payload = (event.payload ?? {}) as { text?: string; message?: string };
          if (event.kind === "text" || event.kind === "done") {
            text = payload.text ?? text;
            if (event.kind === "text") setReply(text);
          } else if (event.kind === "error") failed = payload.message ?? "The reply failed.";
        },
        ctl.signal,
      );
    } catch (e) {
      failed = errorText(e);
    }
    if (ctl.signal.aborted) return;
    if (failed) setError(failed);
    else if (text) setThread([...messages, { role: "assistant", content: text }]);
    setReply(null);
  }

  return (
    <Card data-testid="kept-card">
      <CardHeader>
        <CardTitle className="text-xs font-medium uppercase tracking-wide text-highlight">Round {round} · kept, not live</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <p className="text-sm text-muted-foreground">
          This round failed a check, so Chat still uses the previous checkpoint. Try the new one here, and adopt it only if you prefer it. Adopting
          makes it version {version ?? "—"}&apos;s checkpoint, and the next rounds start from it.
        </p>
        {thread.length > 0 || reply !== null ? (
          <ul className="flex flex-col gap-2" aria-label={`Chat with Round ${round}'s checkpoint`}>
            {thread.map((m, i) => (
              <li key={i} className="text-sm">
                <span className="mr-2 text-xs uppercase text-muted-foreground">{m.role === "user" ? "You" : "Round " + round}</span>
                {m.content}
              </li>
            ))}
            {reply !== null && (
              <li className="text-sm">
                <span className="mr-2 text-xs uppercase text-muted-foreground">Round {round}</span>
                {reply || "…"}
              </li>
            )}
          </ul>
        ) : null}
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void send();
          }}
        >
          <Input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Ask the kept checkpoint something" aria-label={`Message for Round ${round}'s checkpoint`} />
          <Button type="submit" variant="outline" disabled={!draft.trim() || reply !== null}>
            Try in chat
          </Button>
        </form>
        <div>
          <Button onClick={() => adopt.mutate()} disabled={adopt.isPending}>
            Adopt this checkpoint
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
