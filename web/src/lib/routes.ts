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
export type Route = { route: string; href: string; area: string; title: string; lead: string; lookInside?: boolean; hidden?: boolean };

export const ROUTES: Route[] = [
  { route: "builder", href: "/builder", area: "models", title: "Build your model", lead: "Change the picture. See what it costs and what it does." },
  { route: "quicktest", href: "/quick-test", area: "models", title: "Quick test", lead: "Try a design idea for a few minutes on a tiny dataset, before you save it and train it for real." },
  { route: "compare", href: "/compare", area: "models", title: "Compare runs", lead: "Change one thing, hold the rest equal, run several seeds." },
  { route: "trace", href: "/trace", area: "models", title: "Trace a position", lead: "Follow one position's stream through the model.", lookInside: true },
  { route: "head_zoom", href: "/head-zoom", area: "models", title: "Inside a head", lead: "One attention head, one step at a time, on a real prompt.", lookInside: true },
  { route: "value_map", href: "/value-map", area: "models", title: "Value map", lead: "How a head's values are projected down and back up, and why the heads add up.", lookInside: true },
  { route: "loss", href: "/loss", area: "models", title: "Loss", lead: "What one wrong or right prediction costs, and the gradient it sends back.", lookInside: true },
  { route: "landscape", href: "/landscape", area: "models", title: "Loss landscape", lead: "The error as you move the weights, drawn live while a version trains.", lookInside: true },
  { route: "head_lab", href: "/head-lab", area: "models", title: "Head lab", lead: "Where does each attention head look, and how much does the model rely on it?", lookInside: true },
  { route: "inside3d", href: "/inside-3d", area: "models", title: "Inside the network", lead: "Build the model from the inside and watch it train.", lookInside: true },
  { route: "your_data", href: "/your-data", area: "data", title: "Your data", lead: "Three kinds, because each one trains a different stage." },
  { route: "make_data", href: "/make-data", area: "data", title: "Make extra data", lead: "Optional. Generate more, then keep what passes review." },
  { route: "pretraining", href: "/pretraining", area: "data", title: "Pretraining", lead: "Feed it text. It learns by predicting the next word, over and over." },
  { route: "collect", href: "/collect", area: "data", title: "Collect data", lead: "Gather open datasets and question-and-answer data in the background, only from sources whose terms allow it." },
  { route: "storage", href: "/storage", area: "data", title: "Storage", lead: "What the app keeps on this computer's disk, and what you can delete to free space." },
  { route: "runs", href: "/runs", area: "runs", title: "Runs", lead: "Every training run on this computer and the cluster: what it is doing now and how the last ones ended." },
  { route: "new_run", href: "/runs/new", area: "runs", title: "New run", lead: "Pick a model, the data, and the machines to train it on." },
  { route: "run_view", href: "/runs/view", area: "runs", title: "Run", lead: "One run in detail: its rounds, its loss, and each machine's part.", hidden: true },
  { route: "training_run", href: "/training-run", area: "runs", title: "Training run", lead: "Step k of n, the schedule phase and the error so far." },
  { route: "post_training", href: "/post-training", area: "runs", title: "Post-training", lead: "Four rounds, always in this order. Same design throughout; only the weights shift." },
  { route: "queue", href: "/queue", area: "runs", title: "Queue", lead: "What it's training now, what's waiting, and when it's allowed to work." },
  { route: "passive", href: "/passive", area: "runs", title: "Passive training", lead: "Training that keeps going on its own, outside the queue." },
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
