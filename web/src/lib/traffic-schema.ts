// Reads a traffic record's bodies against the server's OpenAPI document (GET /openapi.json, fetched when a row is first expanded,
// not bundled: it is 400 KB). The server names a record's models (studio_api.traffic route_names); this finds them in
// components.schemas, labels each field with its schema type, and flags a body that does not match. Secrets arrive masked (MASK),
// and a masked value is never flagged.

import type { TrafficRecord } from "@/lib/traffic";

export type Schema = {
  $ref?: string;
  type?: string | string[];
  enum?: unknown[];
  const?: unknown;
  properties?: Record<string, Schema>;
  required?: string[];
  items?: Schema;
  anyOf?: Schema[];
  oneOf?: Schema[];
  allOf?: Schema[];
  additionalProperties?: boolean | Schema;
  format?: string;
};

export type Spec = {
  paths?: Record<string, Record<string, { operationId?: string }>>;
  components?: { schemas?: Record<string, Schema> };
};

/** What the server writes in place of a secret (studio_api.traffic.MASK). */
export const MASK = "••••••";
const DEPTH = 8;

export function resolve(schema: Schema | undefined, spec: Spec): Schema | undefined {
  let at = schema;
  for (let hops = 0; at?.$ref && hops < 10; hops++) at = spec.components?.schemas?.[at.$ref.split("/").pop() ?? ""];
  return at;
}

/** The schema of a model name the server sent ("PretrainDistributedRequest"), or undefined for a name the document lacks (list[Foo], dict[...]). */
export function modelSchema(name: string | null, spec: Spec): Schema | undefined {
  return name ? spec.components?.schemas?.[name] : undefined;
}

/** The operationId of a route template and method, e.g. `/collective/{run}/join` and POST. */
export function operationId(record: Pick<TrafficRecord, "route" | "method">, spec: Spec): string | null {
  return (record.route && spec.paths?.[record.route]?.[record.method.toLowerCase()]?.operationId) || null;
}

/** A short type label for a field: "integer", "enum a | b", "list of string", "string | null", or the model's name. */
export function typeLabel(schema: Schema | undefined, spec: Spec, depth = 0): string {
  if (!schema) return "";
  if (schema.$ref) return schema.$ref.split("/").pop() ?? "";
  if (depth > 3) return "object";
  if (schema.enum) return `enum ${schema.enum.map(String).join(" | ")}`;
  if (schema.const !== undefined) return `const ${String(schema.const)}`;
  const choices = schema.anyOf ?? schema.oneOf;
  if (choices) return choices.map((c) => typeLabel(c, spec, depth + 1)).join(" | ");
  if (schema.allOf) return schema.allOf.map((c) => typeLabel(c, spec, depth + 1)).join(" & ");
  const type = Array.isArray(schema.type) ? schema.type.join(" | ") : schema.type;
  if (type === "array") return `list of ${typeLabel(schema.items, spec, depth + 1) || "any"}`;
  return type ?? "";
}

function kind(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value === "number") return Number.isInteger(value) ? "integer" : "number";
  return typeof value;
}

function typeFits(type: string, value: unknown): boolean {
  const actual = kind(value);
  return type === actual || (type === "number" && actual === "integer") || (type === "integer" && actual === "number" && Number.isInteger(value));
}

function check(value: unknown, schema: Schema | undefined, spec: Spec, path: string, out: string[], depth: number): void {
  const at = resolve(schema, spec);
  if (!at || depth > DEPTH || out.length >= 8 || value === MASK) return;
  const where = path || "body";
  const choices = at.anyOf ?? at.oneOf;
  if (choices) {
    const scratch = choices.map((choice) => {
      const found: string[] = [];
      check(value, choice, spec, path, found, depth + 1);
      return found;
    });
    if (!scratch.some((found) => found.length === 0)) out.push(scratch[0]?.[0] ?? `${where}: matches none of the allowed shapes`);
    return;
  }
  if (at.allOf) {
    for (const part of at.allOf) check(value, part, spec, path, out, depth + 1);
    return;
  }
  if (at.enum && !at.enum.some((choice) => choice === value)) {
    out.push(`${where}: ${JSON.stringify(value)} is not one of ${at.enum.map(String).join(", ")}`);
    return;
  }
  if (at.const !== undefined && at.const !== value) {
    out.push(`${where}: expected ${String(at.const)}`);
    return;
  }
  const types = Array.isArray(at.type) ? at.type : at.type ? [at.type] : [];
  if (types.length && !types.some((t) => typeFits(t, value))) {
    out.push(`${where}: expected ${types.join(" or ")}, got ${kind(value)}`);
    return;
  }
  if (Array.isArray(value)) {
    value.slice(0, 20).forEach((item, i) => check(item, at.items, spec, `${path}[${i}]`, out, depth + 1));
  } else if (value && typeof value === "object") {
    const fields = value as Record<string, unknown>;
    for (const name of at.required ?? []) if (!(name in fields)) out.push(`${path ? `${path}.` : ""}${name}: required, missing`);
    for (const [name, field] of Object.entries(fields)) {
      const sub = at.properties?.[name];
      const next = path ? `${path}.${name}` : name;
      if (sub) check(field, sub, spec, next, out, depth + 1);
      else if (at.additionalProperties === false) out.push(`${next}: not a field of this model`);
      else if (typeof at.additionalProperties === "object") check(field, at.additionalProperties, spec, next, out, depth + 1);
    }
  }
}

/** What is wrong with `value` as an instance of `schema`: nothing for a body that fits (an empty list), up to eight lines otherwise. */
export function problems(value: unknown, schema: Schema | undefined, spec: Spec): string[] {
  const out: string[] = [];
  check(value, schema, spec, "", out, 0);
  return out;
}

export type Field = { name: string; value: unknown; type: string };

/** The fields of an object body in the order the model declares them (then any others), each with its schema type. */
export function fields(value: Record<string, unknown>, schema: Schema | undefined, spec: Spec): Field[] {
  const at = resolve(schema, spec);
  const declared = Object.keys(at?.properties ?? {}).filter((name) => name in value);
  const rest = Object.keys(value).filter((name) => !declared.includes(name));
  return [...declared, ...rest].map((name) => ({ name, value: value[name], type: typeLabel(at?.properties?.[name], spec) || kind(value[name]) }));
}
