import { describe, expect, test } from "vitest";
import { newestFirst, summary, toggled, type Lesson } from "./lessons";

const lesson = (id: string, status: Lesson["status"]): Lesson => ({ id, round: 1, checkpoint: 1, prompt: "", reply: "", answer: "", status });

describe("lessons", () => {
  test("a used lesson is dropped, any other is kept again", () => {
    expect(toggled(lesson("a", "pending"))).toEqual({ status: "dropped", label: "Drop" });
    expect(toggled(lesson("a", "dropped"))).toEqual({ status: "pending", label: "Keep" });
    expect(toggled(lesson("a", "left_out"))).toEqual({ status: "pending", label: "Keep" });
  });

  test("the summary counts the lessons in use", () => {
    expect(summary([])).toBe("0 lessons, 0 used in training");
    expect(summary([lesson("a", "pending")])).toBe("1 lesson, 1 used in training");
    expect(summary([lesson("a", "pending"), lesson("b", "dropped"), lesson("c", "left_out")])).toBe("3 lessons, 1 used in training");
  });

  test("newest first does not reorder the original", () => {
    const list = [lesson("a", "pending"), lesson("b", "pending")];
    expect(newestFirst(list).map((l) => l.id)).toEqual(["b", "a"]);
    expect(list.map((l) => l.id)).toEqual(["a", "b"]);
  });
});
