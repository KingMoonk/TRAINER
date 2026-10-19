"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { api, unwrap } from "@/lib/api";
import { problem } from "@/lib/cluster";
import { PASSIVE } from "./copy";
import { LESSON_STATUS, newestFirst, summary, toggled } from "./lessons";

/** What the teacher showed the model in this session, with a Drop or Keep button for each: a dropped lesson is left out of the next round's training. */
export function Lessons({ id }: { id: string }) {
  const client = useQueryClient();
  const key = ["passive", "lessons", id];
  const lessons = useQuery({ queryKey: key, queryFn: () => unwrap(api.GET("/passive/{session_id}/lessons", { params: { path: { session_id: id } } })) });
  const change = useMutation({
    mutationFn: ({ lesson, status }: { lesson: string; status: "pending" | "dropped" }) =>
      unwrap(api.PATCH("/passive/{session_id}/lessons/{lesson_id}", { params: { path: { session_id: id, lesson_id: lesson } }, body: { status } })),
    onError: (error) => toast.error(`${PASSIVE.actionFailed}: ${problem(error)}`),
    onSettled: () => client.invalidateQueries({ queryKey: key }),
  });

  return (
    <section className="flex flex-col gap-2" aria-label={PASSIVE.lessonsTitle}>
      <h3 className="text-sm font-medium">{PASSIVE.lessonsTitle}</h3>
      {lessons.isError ? (
        <p role="alert" className="text-sm text-destructive">
          {PASSIVE.lessonsFailed} {problem(lessons.error)}
        </p>
      ) : !lessons.data ? null : lessons.data.length === 0 ? (
        <p className="text-sm text-muted-foreground">{PASSIVE.lessonsNone}</p>
      ) : (
        <>
          <p className="text-xs text-muted-foreground" data-testid="lessons-summary">
            {summary(lessons.data)}. {PASSIVE.lessonsHint}
          </p>
          <ul className="flex flex-col gap-2" data-testid="passive-lessons">
            {newestFirst(lessons.data).map((lesson) => {
              const next = toggled(lesson);
              return (
                <li key={lesson.id} data-testid={`lesson-${lesson.id}`} data-status={lesson.status} className="flex items-start justify-between gap-4 rounded-md border p-3 text-sm">
                  <div className="flex min-w-0 flex-col gap-1">
                    <span className="text-xs text-muted-foreground">
                      {PASSIVE.lessonFrom(lesson.round, lesson.checkpoint)} · {LESSON_STATUS[lesson.status]}
                    </span>
                    <span className="break-words whitespace-pre-wrap">
                      <span className="text-muted-foreground">{PASSIVE.lessonPrompt} </span>
                      {lesson.prompt}
                    </span>
                    <span className="break-words whitespace-pre-wrap">
                      <span className="text-muted-foreground">{PASSIVE.lessonReply} </span>
                      {lesson.reply}
                    </span>
                    <span className="break-words whitespace-pre-wrap">
                      <span className="text-muted-foreground">{PASSIVE.lessonAnswer} </span>
                      {lesson.answer}
                    </span>
                  </div>
                  <Button variant="outline" size="sm" disabled={change.isPending} onClick={() => change.mutate({ lesson: lesson.id, status: next.status })}>
                    {next.label}
                  </Button>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </section>
  );
}
