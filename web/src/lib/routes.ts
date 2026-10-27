// The app's pages. `route` is the route name; `href` is the URL. The first route of each area is its home.

export type Area = { id: string; title: string };

export const AREAS: Area[] = [
  { id: "models", title: "Models" },
  { id: "data", title: "Data" },
  { id: "train", title: "Train" },
  { id: "eval", title: "Chat" },
  { id: "machines", title: "Machines" },
];

// `hidden` routes are reached from elsewhere (a link from another page) and stay out of the sidebar's page list.
// `parent` is the route whose tab stays active when viewing a hidden route (e.g. `runs` for `run_view`).
export type Route = {
  route: string;
  href: string;
  area: string;
  title: string;
  tabTitle?: string;
  lead: string;
  hidden?: boolean;
  parent?: string;
};

export const ROUTES: Route[] = [
  { route: "builder", href: "/builder", area: "models", title: "Build your model", lead: "Change the picture. See what it costs and what it does." },
  { route: "quicktest", href: "/quick-test", area: "models", title: "Quick test", lead: "Try a design idea for a few minutes on a tiny dataset, before you save it and train it for real.", hidden: true, parent: "builder" },
  { route: "compare", href: "/compare", area: "models", title: "Compare runs", lead: "Change one thing, hold the rest equal, run several seeds.", hidden: true, parent: "builder" },
  { route: "your_data", href: "/your-data", area: "data", title: "Your data", lead: "Three kinds, because each one trains a different stage." },
  { route: "collect", href: "/collect", area: "data", title: "Collect data", lead: "Gather open datasets and question-and-answer data in the background, only from sources whose terms allow it.", hidden: true, parent: "your_data" },
  { route: "storage", href: "/storage", area: "data", title: "Storage", lead: "What the app keeps on this computer's disk, and what you can delete to free space.", hidden: true, parent: "your_data" },
  { route: "auto_run", href: "/auto-run", area: "train", title: "Auto Run", lead: "Plan the whole training once. It runs pretraining and the four rounds in order, and asks you when one fails." },
  { route: "pretraining", href: "/pretraining", area: "train", title: "Pretraining", lead: "Feed it text. It learns by predicting the next word, over and over." },
  { route: "post_training", href: "/post-training", area: "train", title: "Post-training", lead: "Four rounds, always in this order. Same design throughout; only the weights shift." },
  { route: "runs", href: "/runs", area: "train", title: "Runs", tabTitle: "All runs", lead: "Every training run on this computer and the cluster: what it is doing now and how the last ones ended." },
  { route: "new_run", href: "/runs/new", area: "train", title: "New run", lead: "Pick a model, the data, and the machines to train it on.", hidden: true, parent: "runs" },
  { route: "queue", href: "/queue", area: "train", title: "Queue", lead: "What it's training now, what's waiting, and when it's allowed to work.", hidden: true, parent: "runs" },
  { route: "training_run", href: "/training-run", area: "train", title: "Training run", lead: "Step k of n, the schedule phase and the error so far.", hidden: true, parent: "runs" },
  { route: "run_view", href: "/runs/view", area: "train", title: "Run", lead: "One run in detail: its rounds, its loss, and each machine's part.", hidden: true, parent: "runs" },
  { route: "chat", href: "/chat", area: "eval", title: "Chat", lead: "Talk to any trained version. Compare two to see what a change really did." },
  { route: "benchmark", href: "/benchmark", area: "eval", title: "Benchmark", lead: "Score a version on a fixed set of questions." },
  { route: "export", href: "/export", area: "eval", title: "Export", lead: "Write a trained version to a file other tools can open." },
  { route: "machines", href: "/cluster/machines", area: "machines", title: "Machines", lead: "What is running where on the machines this one works with, and who may call this computer." },
  { route: "cluster_training", href: "/cluster/training", area: "machines", title: "Training", tabTitle: "Cluster training", lead: "Start a training run that spreads over several machines, and watch each one: its round, its speed, and why one fell behind." },
];

export const HOME = "/builder";

export function areaHome(areaId: string): Route {
  return ROUTES.find((r) => r.area === areaId)!;
}

export function routeFor(route: string): Route {
  return ROUTES.find((r) => r.route === route)!;
}
