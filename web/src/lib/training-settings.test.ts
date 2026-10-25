import { describe, expect, test } from "vitest";
import type { Choices } from "@/lib/builder";
import { TRAINING_FIELDS, checkSetting, settingValue, showNumber } from "./training-settings";

const field = (key: string) => TRAINING_FIELDS.find((f) => f.key === key)!;

describe("checkSetting", () => {
  test("accepts values in range", () => {
    expect(checkSetting(field("batch_size"), "32")).toEqual({ value: 32 });
    expect(checkSetting(field("learning_rate"), " 0.0005 ")).toEqual({ value: 0.0005 });
    expect(checkSetting(field("dropout"), "0")).toEqual({ value: 0 });
    expect(checkSetting(field("warmup"), "0")).toEqual({ value: 0 });
    expect(checkSetting(field("seed"), "2147483647")).toEqual({ value: 2147483647 });
  });

  test("refuses what the server refuses, with a sentence", () => {
    expect(checkSetting(field("batch_size"), "500")).toEqual({ error: "Batch size must be between 1 and 128." });
    expect(checkSetting(field("batch_size"), "0")).toEqual({ error: "Batch size must be between 1 and 128." });
    expect(checkSetting(field("accumulation"), "65")).toEqual({ error: "Gradient accumulation must be between 1 and 64." });
    expect(checkSetting(field("dropout"), "0.9")).toEqual({ error: "Dropout must be between 0 and 0.5." });
    expect(checkSetting(field("learning_rate"), "5")).toEqual({ error: "Learning rate must be between 0.0000001 and 0.1." });
    expect(checkSetting(field("seed"), "-1")).toEqual({ error: "Seed must be between 0 and 2147483647." });
  });

  test("refuses text that is not a number, and a fraction where a whole number is needed", () => {
    expect(checkSetting(field("batch_size"), "abc")).toEqual({ error: "Batch size must be a number." });
    expect(checkSetting(field("batch_size"), "")).toEqual({ error: "Batch size must be a number." });
    expect(checkSetting(field("batch_size"), "2.5")).toEqual({ error: "Batch size must be a whole number." });
    expect(checkSetting(field("eval_every"), "Infinity")).toEqual({ error: "Evaluate every (steps) must be a number." });
  });
});

describe("settingValue and showNumber", () => {
  test("a setting the design does not have is null", () => {
    expect(settingValue({ warmup: null, batch_size: 4 } as unknown as Choices, "warmup")).toBeNull();
    expect(settingValue({ warmup: 20 } as unknown as Choices, "warmup")).toBe(20);
  });

  test("numbers are shown without float noise", () => {
    expect(showNumber(0.1 + 0.2)).toBe("0.3");
    expect(showNumber(0.0005)).toBe("0.0005");
    expect(showNumber(32)).toBe("32");
  });
});
