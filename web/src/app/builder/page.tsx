import { VersionTray } from "./version-tray";
import { routeFor } from "@/lib/routes";
import { T } from "./copy";

export default function Page() {
  const route = routeFor("builder");
  return <main className="mx-auto flex w-full max-w-6xl min-w-0 flex-col gap-5 p-4 md:p-8">
    <h1 className="text-2xl font-semibold tracking-tight">{route.title}</h1>
    <p className="max-w-prose text-muted-foreground">{T.lead}</p>
    <p className="text-sm text-muted-foreground">{T.desktopEditor}</p>
    <VersionTray />
  </main>;
}
