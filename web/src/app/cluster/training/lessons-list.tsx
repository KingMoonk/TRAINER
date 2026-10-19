"use client";

import { Button } from "@/components/ui/button";
import { problem } from "@/lib/cluster";
import { TRAINING as T } from "@/lib/copy/training";
import { lessonAction, lessonFrom, type Lesson, type Run } from "./logic";
import { useChangeLesson, useLessons } from "./use-training";

function Answers({ lesson }: { lesson: Lesson }) {
  if (lesson.items.length === 0) return null;
  return (
    <details>
      <summary className="cursor-pointer text-xs text-muted-foreground">{T.lessonShow}</summary>
      <ul className="mt-2 flex flex-col gap-2">
        {lesson.items.map((item, index) => (
          <li key={index} className="flex flex-col gap-1 border-l border-border pl-3">
            <span className="break-words whitespace-pre-wrap">
              <span className="text-muted-foreground">{T.lessonPrompt} </span>
              {item.prompt}
            </span>
            <span className="break-words whitespace-pre-wrap">
              <span className="text-muted-foreground">{T.lessonReply} </span>
              {item.reply}
            </span>
            <span className="break-words whitespace-pre-wrap">
              <span className="text-muted-foreground">{T.lessonBetter} </span>
              {item.answer}
            </span>
          </li>
        ))}
        {lesson.more > 0 && <li className="text-xs text-muted-foreground">{T.lessonMore.replace("{n}", String(lesson.more))}</li>}
      </ul>
    </details>
  );
}

/**
 * What the teacher gave a run that is still going, one lesson per polish, with a Drop or Keep button on each: a dropped lesson is left
 * out of the rounds that open from now on. Shown only while the run is going, since only its coordinator can change it.
 */
export function Lessons({ run }: { run: Run }) {
  const lessons = useLessons(run.id, true);
  const change = useChangeLesson(run.id);
  if (lessons.isError) {
    return (
      <p role="alert" className="text-sm text-destructive">
        {T.lessonsFailed} {problem(lessons.error)}
      </p>
    );
  }
  if (!lessons.data || lessons.data.length === 0) return null;
  return (
    <section className="flex flex-col gap-2" aria-label={T.lessonsTitle}>
      <h3 className="text-sm font-medium">{T.lessonsTitle}</h3>
      <p className="max-w-prose text-xs text-muted-foreground">{T.lessonsHint}</p>
      <ul className="flex flex-col gap-2" data-testid={`lessons-${run.id}`}>
        {[...lessons.data].reverse().map((lesson) => {
          const action = lessonAction(lesson);
          return (
            <li key={lesson.id} data-testid={`lesson-${lesson.id}`} data-status={lesson.status} className="flex items-start justify-between gap-4 rounded-md border p-3 text-sm">
              <div className="flex min-w-0 flex-col gap-1">
                <span className="text-xs text-muted-foreground">
                  {lessonFrom(lesson)} · {T.lessonStatus[lesson.status]}
                </span>
                <Answers lesson={lesson} />
              </div>
              <Button variant="outline" size="sm" disabled={change.isPending} onClick={() => change.mutate({ lesson: lesson.id, verb: action.verb })}>
                {action.label}
              </Button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
