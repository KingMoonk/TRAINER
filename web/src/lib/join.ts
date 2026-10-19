"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api, unwrap } from "@/lib/api";
import type { components } from "@/lib/api-schema";
import { problem } from "@/lib/cluster";
import { JOIN } from "@/lib/copy/join";
import { useEvent } from "@/lib/events";
import { fill } from "@/lib/members";

export type Invite = components["schemas"]["Invite"];
export type PauseRequest = components["schemas"]["PauseRequest"];
export type Invites = components["schemas"]["Invites"];

const INVITES = ["distributed", "invites"] as const;
const BUSY_MS = 2_000;
const IDLE_MS = 5_000; // a pause request from another computer arrives without an event on this one

/**
 * The runs other computers added this one to, and the pause requests they sent (GET /distributed/invites). Joining needs no accept
 * step, so this is only ever shown. Re-read on every `queue` and `cluster` event, and on a timer: quickly while there is something
 * to watch, slowly otherwise.
 */
export function useInvites() {
  const client = useQueryClient();
  const refresh = () => void client.invalidateQueries({ queryKey: INVITES });
  useEvent("queue", refresh);
  useEvent("cluster", refresh);
  return useQuery({
    queryKey: INVITES,
    queryFn: ({ signal }) => unwrap(api.GET("/distributed/invites", { signal })),
    refetchInterval: (query) => ((query.state.data?.invites.length ?? 0) + (query.state.data?.pause_requests.length ?? 0) > 0 ? BUSY_MS : IDLE_MS),
    retry: false,
  });
}

/** The invite for a queued job (the Queue page's member job), or undefined. */
export function inviteFor(invites: Invites | undefined, jobId: string): Invite | undefined {
  return invites?.invites.find((invite) => invite.job_id === jobId);
}

function useInvitesWrite() {
  const client = useQueryClient();
  return (invites: Invites) => {
    client.setQueryData<Invites>(INVITES, invites);
    void client.invalidateQueries({ queryKey: ["queue"] });
  };
}

/** Pause the job running here and join now (POST /distributed/invites/{id}/pause-and-join). */
export function usePauseAndJoin() {
  const write = useInvitesWrite();
  return useMutation({
    mutationFn: (invite: Invite) => unwrap(api.POST("/distributed/invites/{job_id}/pause-and-join", { params: { path: { job_id: invite.job_id } } })),
    onSuccess: (invites, invite) => {
      write(invites);
      toast.success(fill(JOIN.pausedAndJoining, { label: invite.blocking_job?.label ?? "" }));
    },
    onError: (error) => toast.error(fill(JOIN.couldNotPause, { why: problem(error) })),
  });
}

/** Leave the run (POST /distributed/invites/{id}/leave): the owner is told, and the member job leaves this queue. */
export function useLeave() {
  const write = useInvitesWrite();
  return useMutation({
    mutationFn: (invite: Invite) => unwrap(api.POST("/distributed/invites/{job_id}/leave", { params: { path: { job_id: invite.job_id } } })),
    onSuccess: (invites, invite) => {
      write(invites);
      toast.success(fill(JOIN.left, { run: invite.run_label }));
    },
    onError: (error) => toast.error(fill(JOIN.couldNotLeave, { why: problem(error) })),
  });
}

/** Say yes to a pause request (POST /distributed/pause-requests/{id}/accept): the running job is held. Only a person here does this. */
export function useAcceptPause() {
  const write = useInvitesWrite();
  return useMutation({
    mutationFn: (request: PauseRequest) => unwrap(api.POST("/distributed/pause-requests/{request_id}/accept", { params: { path: { request_id: request.id } } })),
    onSuccess: (invites) => {
      write(invites);
      toast.success(JOIN.pauseAccepted);
    },
    onError: (error) => toast.error(fill(JOIN.couldNotAccept, { why: problem(error) })),
  });
}

/** Say no (DELETE /distributed/pause-requests/{id}). */
export function useDeclinePause() {
  const write = useInvitesWrite();
  return useMutation({
    mutationFn: (request: PauseRequest) => unwrap(api.DELETE("/distributed/pause-requests/{request_id}", { params: { path: { request_id: request.id } } })),
    onSuccess: write,
    onError: (error) => toast.error(problem(error)),
  });
}

/** Ask another machine to pause the job it is running (POST /distributed/peers/pause-request). It shows there as an alert. */
export function useAskPeerToPause() {
  return useMutation({
    mutationFn: (ask: { address: string; node: string; label: string; runLabel?: string }) =>
      unwrap(api.POST("/distributed/peers/pause-request", { body: { address: ask.address, run_label: ask.runLabel ?? "" } })),
    onSuccess: (_request, ask) => toast.success(fill(ask.label ? JOIN.asked : JOIN.askedNoLabel, { node: ask.node, label: ask.label })),
    onError: (error, ask) => toast.error(fill(JOIN.couldNotAsk, { node: ask.node, why: problem(error) })),
  });
}
