import type { Gate } from "./logic";

/** What a round's gate record carries beyond the verdict (studio_ia/services.py record_round, adopt_round). */
export type KeptGate = Gate & { checkpoint?: string | null; adopted?: boolean };

/** A round that failed its gate and left a checkpoint that is neither live nor adopted: the one that can be tried or adopted. */
export function isKept(gate: KeptGate | undefined): boolean {
  return !!gate && gate.ok === false && !gate.adopted && !!gate.checkpoint;
}

/**
 * Why a Round 2-4 on this version would not build on a Round 1 result: Round 1 failed its gate (and its candidate was not adopted)
 * or has not run. Null when round 1 is live or adopted, is queued or running, or the selected round is Round 1.
 */
export function baseWarning(
  round: number,
  gates: Record<string, KeptGate> | undefined,
  round1Status: string | undefined,
  version: number | undefined,
): string | null {
  if (round < 2 || version === undefined) return null;
  if (round1Status === "running" || round1Status === "queued") return null;
  const gate = gates?.["1"];
  if (!gate) return `Round 1 has not run on version ${version}, so Round ${round} starts from its pretrained checkpoint, not from a model that learned the shape of an answer.`;
  if (gate.ok || gate.adopted) return null;
  return `Round 1 failed its checks on version ${version} and was kept, not live, so Round ${round} starts from its pretrained checkpoint. Adopt Round 1's checkpoint first, or run Round 1 again.`;
}
