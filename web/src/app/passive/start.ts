import type { components } from "@/lib/api-schema";

export type StartBody = components["schemas"]["StartPassive"];
export type Unit = "minutes" | "hours" | "days";

/** The New session form as the person typed it: numbers stay text until they are sent. Defaults are studio_passive.DEFAULT_PLAN. */
export type StartForm = {
  version: string; // "" = the current version
  all: boolean; // train on every text set
  sources: string[]; // the sets ticked when `all` is off
  duration: string;
  unit: Unit;
  patience: string; // rounds without a better checkpoint; 0 = never stop for it
  target: string; // "" = no target error
  teacher: boolean;
  memory: string; // percent of memory
  keepAwake: boolean;
  pauseOnBattery: boolean;
};

export const INITIAL_FORM: StartForm = {
  version: "",
  all: true,
  sources: [],
  duration: "8",
  unit: "hours",
  patience: "3",
  target: "",
  teacher: true,
  memory: "50",
  keepAwake: true,
  pauseOnBattery: true,
};

export const UNITS: Unit[] = ["minutes", "hours", "days"];

export function toggled(list: string[], id: string, on: boolean): string[] {
  return on ? (list.includes(id) ? list : [...list, id]) : list.filter((x) => x !== id);
}

const number = (text: string): number | null => (text.trim() === "" || !Number.isFinite(Number(text)) ? null : Number(text));

/** The request the form stands for, or the sentence saying what to fix. The server checks the plan again and answers in words too. */
export function startBody(form: StartForm): { body: StartBody } | { problem: string } {
  const duration = number(form.duration);
  if (duration === null || duration <= 0) return { problem: "The time limit must be a number above zero." };
  const patience = number(form.patience);
  if (patience === null || patience < 0 || !Number.isInteger(patience)) return { problem: "Rounds without a better checkpoint must be a whole number, 0 or more." };
  const target = form.target.trim() === "" ? null : number(form.target);
  if (form.target.trim() !== "" && (target === null || target < 0)) return { problem: "The target test error must be a number, 0 or more, or empty." };
  const memory = number(form.memory);
  if (memory === null || memory <= 0 || memory > 100) return { problem: "The memory share must be a percentage above 0 and up to 100." };
  if (!form.all && form.sources.length === 0) return { problem: "Tick at least one set of text, or train on all of it." };
  return {
    body: {
      version: form.version === "" ? null : Number(form.version),
      sources: form.all ? null : form.sources,
      duration,
      unit: form.unit,
      patience_rounds: patience,
      target_error: target,
      teacher: form.teacher,
      memory_fraction: memory / 100,
      keep_awake: form.keepAwake,
      pause_on_battery: form.pauseOnBattery,
    },
  };
}
