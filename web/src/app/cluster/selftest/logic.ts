import type { components } from "@/lib/api-schema";
import { SELFTEST as T } from "./copy";

export type SelfTestRun = components["schemas"]["SelfTestRun"];
export type SelfTestStep = components["schemas"]["SelfTestStep"];

/** The headline for a test in its state. */
export function headline(run: SelfTestRun | undefined): string {
  if (!run || run.state === "never") return T.never;
  if (run.state === "running") return T.inProgress;
  return run.state === "passed" ? T.passed : T.failed;
}

/** "Last run 3 Mar 14:05, with box and mac": empty until a test has ended. */
export function ranAt(run: SelfTestRun | undefined, when: (seconds: number) => string): string {
  if (!run || run.finished_at === null || run.state === "never") return "";
  return T.ranAt.replace("{when}", when(run.finished_at)).replace("{machines}", run.machines.join(", "));
}

/** The first failed step: the one whose fix to show first. */
export function firstFailure(run: SelfTestRun | undefined): SelfTestStep | undefined {
  return run?.steps.find((step) => step.status === "fail");
}

/** The steps to list: every step in order (the server lists all of them from the start), none before a test has run. */
export function visible(run: SelfTestRun | undefined): SelfTestStep[] {
  return run && run.state !== "never" ? run.steps : [];
}
