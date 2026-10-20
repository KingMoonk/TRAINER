"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap } from "@/lib/api";
import { problem } from "@/lib/cluster";

const KEY = ["cluster", "selftest"] as const;

/** The cluster self-test (GET /cluster/selftest), re-read every second while it runs. */
export function useSelfTest() {
  return useQuery({
    queryKey: KEY,
    queryFn: ({ signal }) => unwrap(api.GET("/cluster/selftest", { signal })),
    refetchInterval: (query) => (query.state.data?.state === "running" ? 1_000 : false),
    retry: false,
  });
}

export function useStartSelfTest(onProblem: (why: string) => void) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (machines: string[]) => unwrap(api.POST("/cluster/selftest", { body: { machines } })),
    onSuccess: (run) => {
      client.setQueryData(KEY, run);
      void client.invalidateQueries({ queryKey: KEY });
    },
    onError: (error) => onProblem(problem(error)),
  });
}
