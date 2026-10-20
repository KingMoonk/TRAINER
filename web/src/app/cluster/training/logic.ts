// The Training tab's rules, apart from the page so a test can check them without a browser.
import type { components } from "@/lib/api-schema";
import { TRAINING as T } from "@/lib/copy/training";
import type { Mark, Series } from "@/components/series-chart";

export type Run = components["schemas"]["Run"];
export type Member = components["schemas"]["Member"];
export type Round = components["schemas"]["Round"];
export type Defaults = components["schemas"]["Defaults"];
export type StartRun = components["schemas"]["StartRun"];
export type TextSet = components["schemas"]["TextSet"];
export type ChatSet = components["schemas"]["ChatSet"];
export type ClusterNode = components["schemas"]["ClusterNode"];
export type Lesson = components["schemas"]["Lesson"];

/** The button a lesson offers: a lesson in use can be dropped, any other is kept again. */
export function lessonAction(lesson: Lesson): { verb: "drop" | "keep"; label: string } {
  return lesson.status === "live" ? { verb: "drop", label: T.lessonDrop } : { verb: "keep", label: T.lessonKeep };
}

/** "After round 2 · 3 answers": where a lesson came from and how much it holds (rounds are counted from 1 on the page). */
export function lessonFrom(lesson: Lesson): string {
  return T.lessonFrom.replace("{round}", String(lesson.round + 1)).replace("{rows}", String(lesson.rows)).replace("{answers}", lesson.rows === 1 ? T.lessonAnswer : T.lessonAnswers);
}

/** The number formats a change can travel in, smallest first. */
export const DELTA_DTYPES = ["int8", "float16", "float32"] as const;
export type DeltaDtype = (typeof DELTA_DTYPES)[number];

/** The server's default number format when it is one the form offers, else int8. */
export function deltaDtypeOf(text: string | undefined): DeltaDtype {
  return DELTA_DTYPES.find((dtype) => dtype === text) ?? "int8";
}

/** The form as typed: numbers stay text so a half-typed "0." is not rewritten under the cursor. */
export type Form = { rounds: string; innerSteps: string; outerLr: string; outerMomentum: string; version: string; datasets: string[]; chats: string[]; machines: string[]; minMachines: string; maxMachines: string; overlap: boolean; timeBox: boolean; deltaDtype: DeltaDtype; reviewedOnly: boolean; coordinator: string; teacher: string; polishEvery: string; targetWinRate: string };

/** A run that is still going: the list is re-read every couple of seconds while one is. */
export function isActive(run: Run): boolean {
  return run.state === "queued" || run.state === "starting" || run.state === "waiting" || run.state === "running";
}

/** The form's first values: the server's defaults, the newest design, and the text sets that are switched on. */
export function initialForm(defaults: Defaults | undefined, sets: TextSet[] | undefined): Form {
  return {
    rounds: String(defaults?.rounds ?? ""),
    innerSteps: String(defaults?.inner_steps ?? ""),
    outerLr: String(defaults?.outer_lr ?? ""),
    outerMomentum: String(defaults?.outer_momentum ?? ""),
    version: "",
    datasets: (sets ?? []).filter((set) => set.on).map((set) => set.id),
    chats: [],
    machines: [],
    minMachines: "",
    maxMachines: "",
    overlap: true,
    timeBox: true,
    deltaDtype: deltaDtypeOf(defaults?.delta_dtype),
    reviewedOnly: false,
    coordinator: "",
    teacher: "",
    polishEvery: "",
    targetWinRate: "",
  };
}

/** A whole number of at least one, or null. */
function whole(text: string): number | null {
  const n = Number(text);
  return text.trim() !== "" && Number.isInteger(n) && n >= 1 ? n : null;
}

/** A blank (no limit) is `undefined`; a whole number from one to 64 is itself; anything else is null, which keeps the form incomplete. */
function limit(text: string): number | undefined | null {
  if (text.trim() === "") return undefined;
  const n = whole(text);
  return n !== null && n <= 64 ? n : null;
}

/** A number above zero, or null. */
function positive(text: string): number | null {
  const n = Number(text);
  return text.trim() !== "" && Number.isFinite(n) && n > 0 ? n : null;
}

/** Why the teacher options cannot be started with, in a sentence, or "" when they can (or are not used). Every case that keeps Start off says so here. */
export function teacherProblem(form: Form): string {
  const polishBlank = form.polishEvery.trim() === "";
  const winBlank = form.targetWinRate.trim() === "";
  if (polishBlank && winBlank) return form.teacher === "" ? "" : T.teacherNoUse;
  if (!polishBlank) {
    const polish = whole(form.polishEvery);
    if (polish === null || polish < 1) return T.polishBad;
    const rounds = whole(form.rounds);
    if (rounds !== null && polish >= rounds) return T.polishTooLate;
  }
  if (!winBlank) {
    const percent = Number(form.targetWinRate);
    if (!(percent >= 1 && percent <= 100)) return T.winBad;
  }
  if (form.teacher === "") return T.teacherNeeded;
  if (form.coordinator !== "" && form.machines.includes(form.coordinator)) return T.teacherCoordinator;
  return "";
}

/** The teacher's part of the request: nothing while the options are blank. Only called when `teacherProblem` is empty. */
function teaching(form: Form): Pick<StartRun, "teacher" | "polish_every" | "target_win_rate"> | undefined {
  const polish = whole(form.polishEvery);
  const win = form.targetWinRate.trim();
  if (polish === null && win === "") return undefined;
  return { teacher: form.teacher, ...(polish !== null ? { polish_every: polish } : {}), ...(win !== "" ? { target_win_rate: Number(win) / 100 } : {}) };
}

/** The request the form makes, or null while it is not complete (a setting is not a number, or no text is ticked). */
export function startBody(form: Form): StartRun | null {
  const rounds = whole(form.rounds);
  const innerSteps = whole(form.innerSteps);
  const outerLr = positive(form.outerLr);
  const outerMomentum = Number(form.outerMomentum);
  if (rounds === null || innerSteps === null || outerLr === null) return null;
  if (form.outerMomentum.trim() === "" || !Number.isFinite(outerMomentum) || outerMomentum < 0 || outerMomentum >= 1) return null;
  if (form.datasets.length === 0) return null;
  const least = limit(form.minMachines);
  const most = limit(form.maxMachines);
  if (least === null || most === null) return null;
  if (least !== undefined && most !== undefined && least > most) return null;
  if (teacherProblem(form) !== "") return null;
  const teacher = teaching(form);
  return {
    version: form.version === "" ? null : Number(form.version),
    dataset_ids: teacher ? [...form.datasets, ...form.chats.filter((id) => !form.datasets.includes(id))] : form.datasets, // chat sets only feed the teacher
    machines: form.machines,
    ...(form.coordinator !== "" && form.machines.includes(form.coordinator) ? { coordinator: form.coordinator } : {}),
    rounds,
    inner_steps: innerSteps,
    outer_lr: outerLr,
    outer_momentum: outerMomentum,
    ...(least !== undefined ? { min_members: least } : {}),
    ...(most !== undefined ? { max_members: most } : {}),
    overlap: form.overlap ? null : false, // null lets the server turn it on for a run that spans machines
    time_box: form.timeBox ? null : false, // null lets the server turn it on for a run that spans machines
    delta_dtype: form.deltaDtype,
    reviewed_only: form.reviewedOnly,
    ...teacher,
  };
}

/** The other machines the run can use: the listed ones, answering or not (one that does not answer shows why in the run). */
export function otherMachines(nodes: ClusterNode[] | undefined): ClusterNode[] {
  return (nodes ?? []).filter((node) => !node.local);
}

/** A list with `item` in it when `on`, without it when not. Order is kept. */
export function toggled(list: string[], item: string, on: boolean): string[] {
  return on ? (list.includes(item) ? list : [...list, item]) : list.filter((x) => x !== item);
}

/** "3.8 MB", "512 B", or "" for a size that is not known. */
export function bytesText(bytes: number | null | undefined): string {
  if (bytes == null || bytes < 0) return "";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[unit]}`;
}

/** A member's speed in words, or "" before it has finished a round. */
export function speedText(member: Member): string {
  return member.tokens_per_second == null ? "" : T.tokensPerSecond.replace("{n}", Math.round(member.tokens_per_second).toLocaleString("en-US"));
}

/** What a member's last change weighed, with the number format it was sent in ("3.8 MB, float16"), or "". */
export function deltaText(member: Member): string {
  const size = bytesText(member.last_delta_bytes);
  return size ? [size, member.last_delta_dtype].filter(Boolean).join(", ") : "";
}

/** The machine a member runs on: its device ("RTX 4090 (cuda)"), or "" before the machine said. Never assumes a kind of device. */
export function memberDevice(member: Member): string {
  if (!member.device) return "";
  return member.device_name ? `${member.device_name} (${member.device})` : member.device;
}

/** Whether a member never took part in the run, which has ended: it got no slot, or was invited and never joined. */
export function neverJoined(member: Member): boolean {
  return member.state === "never_joined";
}

/** Whether a member's row shows a problem (red): it is out of the run or late. A machine that never joined is no problem; it only says why. */
export function isProblem(member: Member): boolean {
  return member.state === "unreachable" || member.state === "dropped" || member.state === "behind" || ((member.state === "left" || member.state === "stopped") && member.note !== "");
}

/** Whether a member's row needs a reason shown under it. A waiting machine says what it waits for. */
export function needsReason(member: Member): boolean {
  return neverJoined(member) || isProblem(member) || member.state === "waiting";
}

/** One sentence on why a member is not training, from what the server said, else a plain fallback. */
export function reasonText(member: Member): string {
  if (member.reason || member.note) return member.reason || member.note;
  if (neverJoined(member)) return T.neverJoinedFallback;
  if (member.state === "dropped") return T.droppedFallback;
  if (member.state === "behind") return T.behindFallback;
  return T.unreachableFallback;
}

/** The members that are out of the run or late, so the page can say how many without counting in the markup. */
export function trouble(run: Run): Member[] {
  return run.members.filter(needsReason);
}

/** Loss over the rounds, one line for the training loss and one for the validation loss; a round without a number is left out. */
export function lossSeries(history: Round[]): Series[] {
  const line = (id: string, label: string, pick: (row: Round) => number | null): Series => ({
    id, label, points: history.flatMap((row) => (pick(row) == null ? [] : [{ x: row.round + 1, y: pick(row) as number }])),
  });
  return [line("loss", T.trainLoss, (row) => row.loss), line("val_loss", T.validationLoss, (row) => row.val_loss)].filter((s) => s.points.length > 0);
}

/** The marks on the error curve: the best round (lowest validation loss so far) and each round after which the stop rules changed course. */
export function errorMarks(run: Run): Mark[] {
  const marks: Mark[] = run.history.flatMap((row) =>
    row.pivot ? [{ id: `pivot-${row.round}`, x: row.round + 1, label: T.pivots[row.pivot] ?? T.pivotFallback, dash: "4 3" }] : [],
  );
  const polishColor: Record<string, string> = { accepted: "var(--chart-3)", rejected: "var(--chart-4)", skipped: "var(--muted-foreground)" };
  for (const row of run.history) {
    if (row.polish) marks.push({ id: `polish-${row.round}`, x: row.round + 1, label: T.polishKinds[row.polish], color: polishColor[row.polish], dash: "1 3" });
  }
  if (run.best_round != null) marks.push({ id: "best", x: run.best_round + 1, label: T.bestRound, color: "var(--chart-2)" });
  return marks;
}

/** Which of Pause, Resume and Restart the run offers right now: only the ones the server says it can do. */
export function holdButtons(run: Run): { pause: boolean; resume: boolean; restart: boolean } {
  return { pause: run.can_pause && !run.paused, resume: run.can_resume && run.paused, restart: run.can_restart };
}

/** Why the run ended, in a sentence, or "" while it goes. */
export function stopReasonText(run: Run): string {
  return run.stop_reason ? T.stopReason.replace("{why}", run.stop_reason) : "";
}

/** Bytes moved each round, up to the coordinator and back down, in megabytes so the axis reads as small numbers. */
export function trafficSeries(history: Round[]): Series[] {
  const mb = (bytes: number) => Math.round((bytes / 2 ** 20) * 100) / 100;
  return [
    { id: "up", label: T.up, points: history.map((row) => ({ x: row.round + 1, y: mb(row.bytes_up) })) },
    { id: "down", label: T.down, points: history.map((row) => ({ x: row.round + 1, y: mb(row.bytes_down) })) },
  ];
}

/** What the teacher added to the training text, or nothing when it added none. */
export function lessonsText(run: Run): string {
  if (!run.lessons && !run.lessons_left_out) return "";
  const live = T.lessonsLine.replace("{live}", String(run.lessons)).replace("{lessons}", run.lessons === 1 ? T.lesson : T.lessonsPlural);
  return run.lessons_left_out ? `${live}, ${T.lessonsLeftOut.replace("{out}", String(run.lessons_left_out))}` : live;
}
