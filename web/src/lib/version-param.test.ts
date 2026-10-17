import { renderHook } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { nav } from "@/test/navigation";
import { useVersionParameter } from "./version-param";

function read(search: string): number | undefined {
  nav.params = new URLSearchParams(search);
  return renderHook(() => useVersionParameter()).result.current;
}

describe("useVersionParameter", () => {
  test("reads a positive whole number", () => {
    expect(read("version=3")).toBe(3);
    expect(read("version=12&other=x")).toBe(12);
  });

  test("is undefined without the parameter or with an empty one", () => {
    expect(read("")).toBeUndefined();
    expect(read("version=")).toBeUndefined();
    expect(read("other=3")).toBeUndefined();
  });

  test.each(["0", "-1", "+2", "01", "1.5", "1e3", "abc", " 3", "3 ", "0x10", "3a"])("rejects %j", (value) => {
    expect(read(`version=${encodeURIComponent(value)}`)).toBeUndefined();
  });

  test("accepts the largest safe integer and rejects the next one", () => {
    expect(read("version=9007199254740991")).toBe(Number.MAX_SAFE_INTEGER);
    expect(read("version=9007199254740992")).toBeUndefined();
  });
});
