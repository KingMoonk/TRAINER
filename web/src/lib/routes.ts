// The app's pages, mirroring the Tk shell (studio_ia/shell.py: AREAS, AREA_HOME, ROUTES) until Phase 5 removes it.
// `route` is the Tk route name; `href` is the URL. The first route of each area is its home.

export type Area = { id: string; title: string };

export const AREAS: Area[] = [
  { id: "models", title: "Models" },
  { id: "data", title: "Data" },
  { id: "runs", title: "Runs" },
  { id: "eval", title: "Eval" },
  { id: "machines", title: "Machines" },
];

// `hidden` routes are reached from elsewhere (a link from a run) and stay out of the sidebar's page list.
export type Route = { route: string; href: string; area: string; title: string; lead: string; hidden?: boolean };

export const ROUTES: Route[] = [
  { route: "builder", href: "/builder", area: "models", title: "Build your model", lead: "Change the picture. See what it costs and what it does." },
  { route: "quicktest", href: "/quick-test", area: "models", title: "Quick test", lead: "Try a design idea for a few minutes on a tiny dataset, before you save it and train it for real." },
  { route: "compare", href: "/compare", area: "models", title: "Compare runs", lead: "Change one thing, hold the rest equal, run several seeds." },
  { route: "your_data", href: "/your-data", area: "data", title: "Your data", lead: "Three kinds, because each one trains a different stage." },
  { route: "pretraining", href: "/pretraining", area: "data", title: "Pretraining", lead: "Feed it text. It learns by predicting the next word, over and over." },
  { route: "collect", href: "/collect", area: "data", title: "Collect data", lead: "Gather open datasets and question-and-answer data in the background, only from sources whose terms allow it." },
  { route: "storage", href: "/storage", area: "data", title: "Storage", lead: "What the app keeps on this computer's disk, and what you can delete to free space." },
  { route: "runs", href: "/runs", area: "runs", title: "Runs", lead: "Every training run on this computer and the cluster: what it is doing now and how the last ones ended." },
  { route: "new_run", href: "/runs/new", area: "runs", title: "New run", lead: "Pick a model, the data, and the machines to train it on." },
  { route: "run_view", href: "/runs/view", area: "runs", title: "Run", lead: "One run in detail: its rounds, its loss, and each machine's part.", hidden: true },
  { route: "training_run", href: "/training-run", area: "runs", title: "Training run", lead: "Step k of n, the schedule phase and the error so far." },
  { route: "post_training", href: "/post-training", area: "runs", title: "Post-training", lead: "Four rounds, always in this order. Same design throughout; only the weights shift." },
  { route: "queue", href: "/queue", area: "runs", title: "Queue", lead: "What it's training now, what's waiting, and when it's allowed to work." },
  { route: "cluster_training", href: "/cluster/training", area: "runs", title: "Training", lead: "Start a training run that spreads over several machines, and watch each one: its round, its speed, and why one fell behind." },
  { route: "chat", href: "/chat", area: "eval", title: "Chat", lead: "Talk to any trained version. Compare two to see what a change really did." },
  { route: "benchmark", href: "/benchmark", area: "eval", title: "Benchmark", lead: "Score a version on a fixed set of questions." },
  { route: "machines", href: "/cluster/machines", area: "machines", title: "Machines", lead: "What is running where on the machines this one works with, and who may call this computer." },
];

export const HOME = "/builder";

export function areaHome(areaId: string): Route {
  return ROUTES.find((r) => r.area === areaId)!;
}

export function routeFor(route: string): Route {
  return ROUTES.find((r) => r.route === route)!;
}
