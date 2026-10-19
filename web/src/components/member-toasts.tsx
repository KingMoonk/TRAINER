"use client";

import { useEffect, useRef } from "react";
import { toast } from "sonner";
import { useMachines, useRuns } from "@/app/cluster/training/use-training";
import { useInvites } from "@/lib/join";
import { memberChanges, type Seen } from "@/lib/members";

/**
 * A toast when a machine joins, falls behind, lapses or leaves a run: for the runs this computer coordinates (each other machine) and for
 * the runs it was added to (this machine). Renders nothing. The first read only records, so opening a page toasts nothing.
 */
export function MemberToasts() {
  const runs = useRuns().data;
  const invites = useInvites().data;
  const nodes = useMachines().data;
  const me = nodes?.find((node) => node.local)?.hostname;
  const before = useRef<Map<string, Seen> | null>(null);
  useEffect(() => {
    if (!runs || !invites || !me) return;
    const seen: Seen[] = [
      ...runs
        .filter((run) => run.owner === me)
        .flatMap((run) => run.members.filter((m) => !m.local).map((m) => ({ key: `run:${run.id}:${m.node}`, node: m.node, run: run.label, state: m.state, here: false }))),
      ...invites.invites.map((invite) => ({ key: `invite:${invite.job_id}`, node: invite.node, run: invite.run_label, state: invite.state, here: true })),
    ];
    const { next, changes } = memberChanges(before.current, seen);
    before.current = next;
    for (const change of changes) toast[change.kind](change.text, { id: change.id });
  }, [runs, invites, me]);
  return null;
}
