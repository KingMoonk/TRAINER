import type { components } from "@/lib/api-schema";

export type Lesson = components["schemas"]["PassiveLesson"];

/** How each lesson status reads. */
export const LESSON_STATUS: Record<Lesson["status"], string> = {
  pending: "Used in training",
  dropped: "Dropped by you",
  left_out: "Left out by a step back",
};

/** The next status a lesson's button asks for: a used lesson can be dropped, any other can be kept again. */
export function toggled(lesson: Lesson): { status: "pending" | "dropped"; label: "Drop" | "Keep" } {
  return lesson.status === "pending" ? { status: "dropped", label: "Drop" } : { status: "pending", label: "Keep" };
}

/** Newest lessons first, with how many are in use: the heading line. */
export function summary(lessons: Lesson[]): string {
  const used = lessons.filter((l) => l.status === "pending").length;
  return `${lessons.length} ${lessons.length === 1 ? "lesson" : "lessons"}, ${used} used in training`;
}

export function newestFirst(lessons: Lesson[]): Lesson[] {
  return [...lessons].reverse();
}
