"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, unwrap } from "@/lib/api";
import type { components } from "@/lib/api-schema";
import { awaitJob } from "@/lib/job-result";
import { CollectionCard } from "../your-data/collection-card";
import { useApiEvents } from "../your-data/events";
import { errorText } from "../your-data/logic";
import type { Kind } from "../your-data/copy";
import { MAKE as T } from "./copy";
import { Review, refreshData } from "./review";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";
import { optionBox } from "@/components/option-box";

type Report = components["schemas"]["GenerateReport"];
type Rules = components["schemas"]["ProblemRules"];
const KINDS: Kind[] = ["problems", "conversations", "text"];

export default function Page() {
  return <Suspense fallback={<Skeleton className="m-8 h-48" />}><LinkedGenerator /></Suspense>;
}

function LinkedGenerator() {
  const requested = useSearchParams().get("kind");
  const kind = KINDS.find((kind) => kind === requested) ?? "problems";
  return <MakeData key={kind} initialKind={kind} />;
}

function MakeData({ initialKind }: { initialKind: Kind }) {
  const [kind, setKind] = useState<Kind>(initialKind);
  const [making, setMaking] = useState(false);
  const cache = useQueryClient();
  const rules = useQuery({ queryKey: ["data", "problem-rules"], queryFn: ({ signal }) => unwrap(api.GET("/data/problems/rules", { signal })), staleTime: Infinity });
  const count = useQuery({ queryKey: ["data", "draft-count"], queryFn: ({ signal }) => unwrap(api.GET("/data/drafts", { signal })) });
  useApiEvents((event) => {
    if (event.kind === "data" || event.kind === "downloads_finished") void refreshData(cache);
  });
  return <main className={PAGE_MAIN}>
    <div className="flex flex-wrap items-start justify-between gap-3"><PageHeader title={T.title} lead={T.lead} />
      <Link href="/your-data/" className={buttonVariants({ variant: "outline" })}>{T.yourData}</Link></div>
    <div role="group" aria-label="What to make" className="grid gap-3 md:grid-cols-3">
      {KINDS.map((value) => <button key={value} type="button" disabled={making} aria-pressed={kind === value} onClick={() => setKind(value)}
        className={optionBox(kind === value, "p-4 disabled:opacity-50")}>
        <span className="block font-medium">{T.cards[value].title}</span><span className={`mt-2 block text-xs ${value === "problems" ? "text-info" : "text-warning"}`}>{T.cards[value].safety}</span>
      </button>)}
    </div>
    {rules.isPending ? <Skeleton className="h-48 w-full" aria-label="Loading generation rules" /> : rules.isError ?
      <Card><CardContent role="alert">{errorText(rules.error)} <Button onClick={() => void rules.refetch()}>{T.retry}</Button></CardContent></Card> :
      <Generator key={kind} kind={kind} rules={rules.data} onMaking={setMaking} />}
    {kind !== "problems" && <Review kind={kind} count={count.data?.count} />}
    <CollectionCard kind={kind} description="Recommended sources add real examples alongside the data you make." />
  </main>;
}

function Generator({ kind, rules, onMaking }: { kind: Kind; rules: Rules; onMaking: (value: boolean) => void }) {
  const [amount, setAmount] = useState("20");
  const [rule, setRule] = useState("");
  const controller = useRef<AbortController | null>(null);
  const cache = useQueryClient();
  const max = kind === "problems" ? rules.max_problems : rules.max_drafts;
  const n = Number(amount);
  const valid = /^\d+$/.test(amount) && Number.isSafeInteger(n) && n >= 1 && n <= max;
  useEffect(() => () => controller.current?.abort(), []);
  const make = useMutation({
    mutationFn: async (preview: boolean): Promise<{ report?: Report; count?: number; preview: boolean }> => {
      controller.current = new AbortController();
      const signal = controller.current.signal;
      if (kind === "problems") return { report: await unwrap(api.POST("/data/problems/generate", {
        body: { rule: rule || null, n: preview ? 5 : n, store: !preview }, signal,
      })), preview };
      const initial = await unwrap(api.POST("/data/{kind}/drafts", { params: { path: { kind } }, body: { n }, signal }));
      const result = await awaitJob<{ count: number }>(initial, signal);
      return { count: result.count, preview: false };
    },
    onMutate: () => onMaking(true),
    onSuccess: () => { void refreshData(cache); },
    onSettled: () => onMaking(false),
  });
  return <Card><CardHeader><CardTitle>{T.cards[kind].title}</CardTitle><p className="text-sm text-muted-foreground">{T.cards[kind].note}</p></CardHeader>
    <CardContent className="flex flex-col gap-4">
      <form className="flex flex-wrap items-end gap-3" onSubmit={(event) => { event.preventDefault(); if (valid && !make.isPending) make.mutate(false); }}>
        {kind === "problems" && <label className="flex min-w-0 flex-col gap-1 text-sm">{T.rule}<select aria-label={T.rule} value={rule} disabled={make.isPending} onChange={(event) => setRule(event.target.value)} className="h-10 max-w-full rounded-lg border bg-background px-3">
          <option value="">{T.allRules}</option>{rules.rules.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
        </select></label>}
        <label className="flex flex-col gap-1 text-sm">{T.count}<input aria-label={T.count} inputMode="numeric" type="number" min={1} max={max} value={amount} disabled={make.isPending} onChange={(event) => setAmount(event.target.value)} className="h-10 w-32 rounded-lg border bg-transparent px-3" /></label>
        {kind === "problems" && <Button type="button" variant="outline" disabled={make.isPending} onClick={() => make.mutate(true)}>{T.preview}</Button>}
        <Button type="submit" disabled={!valid || make.isPending}>{kind === "problems" ? T.generate : T.drafts}</Button>
      </form>
      {!valid && <p role="alert" className="text-sm text-destructive">{T.limits(max)}</p>}
      <p className="text-sm text-muted-foreground">{T.cards[kind].where}</p>
      {make.isPending && <div role="status" aria-label={T.generating} className="space-y-2"><p className="text-sm">{T.generating}</p><Skeleton className="h-20 w-full" /></div>}
      {make.isError && <p role="alert" className="text-sm text-destructive">{errorText(make.error)}</p>}
      {make.data && !make.isPending && <div role="status" className="space-y-2 text-sm">
        {make.data.report ? <>
          <p className="font-medium">{make.data.preview ? T.previewTitle : `Added ${make.data.report.added} problems to Your data → Problems (${make.data.report.duplicates} duplicates skipped).`}</p>
          {make.data.preview && <ol className="space-y-2">{make.data.report.items.slice(0, 5).map((item, index) => <li key={index} className="rounded-lg border bg-muted/30 p-3 break-words">{String(item.prompt)} → {String(item.answer ?? "(rule decides)")}</li>)}</ol>}
          <p>{make.data.report.passed} of {make.data.report.checked} sample answers pass their own rule.</p>
        </> : <p>{make.data.count} drafts added for review. Nothing is used until you mark it Good.</p>}
      </div>}
    </CardContent></Card>;
}
