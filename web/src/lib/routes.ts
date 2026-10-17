// The app's pages, mirroring the Tk shell (studio_ia/shell.py: AREAS, AREA_HOME, ROUTES) until Phase 5 removes it.
// `route` is the Tk route name; `href` is the URL. The first route of each area is its home.

export type Area = { id: string; title: string; utility?: boolean };

export const AREAS: Area[] = [
  { id: "designer", title: "Network designer" },
  { id: "pretraining", title: "Pretraining" },
  { id: "post", title: "Post-training" },
  { id: "chat", title: "Chat" },
  { id: "queue", title: "Queue", utility: true },
  { id: "cluster", title: "Cluster", utility: true },
  { id: "resources", title: "Resources", utility: true },
];

export type Route = { route: string; href: string; area: string; title: string; lead: string; lookInside?: boolean };

export const ROUTES: Route[] = [
  { route: "builder", href: "/builder", area: "designer", title: "Build your model", lead: "Change the picture. See what it costs and what it does." },
  { route: "quicktest", href: "/quick-test", area: "designer", title: "Quick test", lead: "Try a design idea for a few minutes on a tiny dataset, before you save it and train it for real." },
  { route: "compare", href: "/compare", area: "designer", title: "Compare runs", lead: "Change one thing, hold the rest equal, run several seeds." },
  { route: "trace", href: "/trace", area: "designer", title: "Trace a position", lead: "Follow one position's stream through the model.", lookInside: true },
  { route: "head_zoom", href: "/head-zoom", area: "designer", title: "Inside a head", lead: "One attention head, one step at a time, on a real prompt.", lookInside: true },
  { route: "value_map", href: "/value-map", area: "designer", title: "Value map", lead: "How a head's values are projected down and back up, and why the heads add up.", lookInside: true },
  { route: "loss", href: "/loss", area: "designer", title: "Loss", lead: "What one wrong or right prediction costs, and the gradient it sends back.", lookInside: true },
  { route: "landscape", href: "/landscape", area: "designer", title: "Loss landscape", lead: "The error as you move the weights, drawn live while a version trains.", lookInside: true },
  { route: "head_lab", href: "/head-lab", area: "designer", title: "Head lab", lead: "Where does each attention head look, and how much does the model rely on it?", lookInside: true },
  { route: "inside3d", href: "/inside-3d", area: "designer", title: "Inside the network", lead: "Build the model from the inside and watch it train.", lookInside: true },
  { route: "pretraining", href: "/pretraining", area: "pretraining", title: "Pretraining", lead: "Feed it text. It learns by predicting the next word, over and over." },
  { route: "training_run", href: "/training-run", area: "pretraining", title: "Training run", lead: "Step k of n, the schedule phase and the error so far." },
  { route: "post_training", href: "/post-training", area: "post", title: "Post-training", lead: "Four rounds, always in this order. Same design throughout; only the weights shift." },
  { route: "chat", href: "/chat", area: "chat", title: "Chat", lead: "Talk to any trained version. Compare two to see what a change really did." },
  { route: "benchmark", href: "/benchmark", area: "chat", title: "Benchmark", lead: "Score a version on a fixed set of questions." },
  { route: "queue", href: "/queue", area: "queue", title: "Queue", lead: "What it's training now, what's waiting, and when it's allowed to work." },
  { route: "cluster", href: "/cluster", area: "cluster", title: "Cluster", lead: "The machines this one works with, what each is doing, and who may call this computer." },
  { route: "your_data", href: "/your-data", area: "resources", title: "Your data", lead: "Three kinds, because each one trains a different stage." },
  { route: "make_data", href: "/make-data", area: "resources", title: "Make extra data", lead: "Optional. Generate more, then keep what passes review." },
  { route: "storage", href: "/storage", area: "resources", title: "Storage", lead: "What the app keeps on this computer's disk, and what you can delete to free space." },
  { route: "passive", href: "/passive", area: "resources", title: "Passive training", lead: "Training that keeps going on its own, outside the queue." },
];

export const HOME = "/builder";

export function areaHome(areaId: string): Route {
  return ROUTES.find((r) => r.area === areaId)!;
}

export function routeFor(route: string): Route {
  return ROUTES.find((r) => r.route === route)!;
}
