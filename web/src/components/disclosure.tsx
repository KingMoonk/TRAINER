"use client";

import { useEffect, useRef } from "react";

/**
 * A native `<details>` for advanced sections. The children stay mounted while it is closed, so their local work (a typed
 * value, a running request) is kept. `defaultOpen` sets the state it starts in; a change in `reveal` to a truthy value opens it.
 */
export function Disclosure({
  title,
  hint,
  defaultOpen,
  reveal,
  testId,
  children,
}: {
  title: string;
  hint?: string;
  defaultOpen?: boolean;
  reveal?: unknown;
  testId?: string;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    if (reveal && ref.current) ref.current.open = true;
  }, [reveal]);
  return (
    <details ref={ref} open={defaultOpen} data-testid={testId} className="min-w-0 rounded-xl border bg-card/50">
      <summary className="cursor-pointer rounded-xl px-4 py-3 text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring">
        {title}
        {hint && <span className="mt-1 block text-xs font-normal text-muted-foreground">{hint}</span>}
      </summary>
      <div className="flex min-w-0 flex-col gap-3 border-t p-3">{children}</div>
    </details>
  );
}
