# Groundwork Studio web UI

The browser front end that replaces the Tk app (docs/rewrite.md). Next.js app router, TypeScript, Tailwind, shadcn/ui,
TanStack Query. Static export: `pnpm build` writes `out/`, which the API server serves.

```bash
pnpm install
pnpm dev            # http://localhost:3000
pnpm lint && pnpm typecheck && pnpm build
```

`NEXT_PUBLIC_API_URL` points `src/lib/api.ts` at the API server; leave it unset when the API serves `out/` itself.
