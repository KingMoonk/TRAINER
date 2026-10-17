import { describe, expect, test } from "vitest";
import { MASK, fields, modelSchema, operationId, problems, typeLabel, type Spec } from "./traffic-schema";

const spec: Spec = {
  paths: { "/collective/{run}/join": { post: { operationId: "join_collective__run__join_post" } } },
  components: {
    schemas: {
      Req: {
        type: "object", required: ["name", "lr"],
        properties: {
          name: { type: "string" }, lr: { type: "number" }, steps: { type: "integer" }, mode: { enum: ["a", "b"] },
          tags: { type: "array", items: { type: "string" } }, opt: { anyOf: [{ type: "integer" }, { type: "null" }] }, inner: { $ref: "#/components/schemas/Inner" },
        },
      },
      Inner: { type: "object", required: ["x"], properties: { x: { type: "integer" } }, additionalProperties: false },
    },
  },
};
const req = modelSchema("Req", spec);

describe("traffic-schema", () => {
  test("finds the model and the operationId", () => {
    expect(req?.required).toEqual(["name", "lr"]);
    expect(modelSchema("list[Req]", spec)).toBeUndefined();
    expect(modelSchema(null, spec)).toBeUndefined();
    expect(operationId({ route: "/collective/{run}/join", method: "POST" }, spec)).toBe("join_collective__run__join_post");
    expect(operationId({ route: "/nope", method: "GET" }, spec)).toBeNull();
  });
  test("a body that fits has no problems", () => {
    expect(problems({ name: "a", lr: 1, steps: 3, mode: "a", tags: ["x"], opt: null, inner: { x: 1 } }, req, spec)).toEqual([]);
  });
  test("flags a missing required field, a wrong type, a bad enum value and a nested mismatch", () => {
    expect(problems({ lr: "fast" }, req, spec)).toEqual(["name: required, missing", "lr: expected number, got string"]);
    expect(problems({ name: "a", lr: 1, mode: "c" }, req, spec)).toEqual(['mode: "c" is not one of a, b']);
    expect(problems({ name: "a", lr: 1, steps: 1.5 }, req, spec)).toEqual(["steps: expected integer, got number"]);
    expect(problems({ name: "a", lr: 1, tags: ["x", 2] }, req, spec)).toEqual(["tags[1]: expected string, got integer"]);
    expect(problems({ name: "a", lr: 1, inner: { y: 1 } }, req, spec)).toEqual(["inner.x: required, missing", "inner.y: not a field of this model"]);
    expect(problems({ name: "a", lr: 1, opt: "s" }, req, spec)).toHaveLength(1);
  });
  test("a masked secret is never flagged", () => {
    expect(problems({ name: MASK, lr: MASK }, req, spec)).toEqual([]);
  });
  test("fields come in declared order with their types", () => {
    const list = fields({ extra: 1, lr: 0.1, name: "a", inner: { x: 1 } }, req, spec);
    expect(list.map((f) => [f.name, f.type])).toEqual([["name", "string"], ["lr", "number"], ["inner", "Inner"], ["extra", "integer"]]);
    expect(typeLabel(req?.properties?.mode, spec)).toBe("enum a | b");
    expect(typeLabel(req?.properties?.tags, spec)).toBe("list of string");
    expect(typeLabel(req?.properties?.opt, spec)).toBe("integer | null");
  });
});
