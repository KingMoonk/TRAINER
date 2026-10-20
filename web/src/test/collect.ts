import type { Config } from "@/app/collect/logic";

// The saved Collect settings the component and unit tests start from.
export const config: Config = {
  enabled: true,
  paused: false,
  topics: ["science", "code"],
  custom_topics: ["astronomy"],
  sources: { hub: true, stackexchange: false, github: false, web: false },
  licences: { non_commercial: false, unknown: false, copyleft: false },
  limits: { disk_mb: 4096, bandwidth_mb_per_day: 2048, requests_per_host_per_day: 5000, seconds_between_requests: 2, max_dataset_mb: 200, max_file_mb: 1024, auto_approve_mb: 100 },
  idle: { only_when_idle: true, only_on_mains: true, unmetered_confirmed: false, cpu_percent: 50 },
  approved: [],
  blocked: [],
  stackexchange: { sites: ["cs.stackexchange.com"], min_score: 3, dump_url: "https://archive.org/download/stackexchange" },
  github: { queries: [], max_repos: 25 },
  web: { sites: [{ host: "docs.example.org", seeds: ["https://docs.example.org/"], licence: "cc-by-4.0", pages: 50 }] },
};
