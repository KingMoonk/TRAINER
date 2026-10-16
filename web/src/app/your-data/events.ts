"use client";

import { useEffect, useRef } from "react";
import { subscribe, type ApiEvent } from "@/lib/api";

// One /events stream shared by everything on the Your data and Pretraining pages (the collection card, the job runner and the
// pages themselves), so a page costs the browser one connection however many parts listen. It opens with the first listener,
// closes with the last and retries while it is down.

const RETRY_MS = 3_000;
const handlers = new Set<(event: ApiEvent) => void>();
let close: (() => void) | null = null;
let retry: ReturnType<typeof setTimeout> | undefined;

function open() {
  retry = undefined;
  close = subscribe(
    (event) => handlers.forEach((handler) => handler(event)),
    () => {
      close?.();
      close = null;
      if (handlers.size) retry = setTimeout(open, RETRY_MS);
    },
  );
}

/** Call `handler` for every event until the returned function is called. */
export function onApiEvent(handler: (event: ApiEvent) => void): () => void {
  handlers.add(handler);
  if (!close && retry === undefined) open();
  return () => {
    handlers.delete(handler);
    if (!handlers.size) {
      close?.();
      close = null;
      clearTimeout(retry);
      retry = undefined;
    }
  };
}

/** Call `handler` (the latest one passed) for every event while the component is mounted. */
export function useApiEvents(handler: (event: ApiEvent) => void) {
  const latest = useRef(handler);
  useEffect(() => {
    latest.current = handler;
  });
  useEffect(() => onApiEvent((event) => latest.current(event)), []);
}
