// The strings this page uses, moved from studio_ia/copy.py (TRAINING) and studio_ia/pages/training_run.py. The Python copy
// stays while the Tk page uses it.

export const T = {
  nothingTraining: "Nothing is training.",
  starting: "Starting",
  noVersion: "No version yet: design one in the Network designer, then train it in Pretraining.",
  seeChange: "see the change",
  pause: "Pause",
  resume: "Resume",
  stopCooldown: "Stop after a short cool-down",
  step: "Step",
  of: "of",
  pass: "Pass",
  overText: "over your text",
  piecesSeen: "pieces seen",
  schedule: "Schedule:",
  lrAuto: "learning rate: auto",
  left: "left",
  coolingDown: "cooling down, then stopping",
  waiting: "waiting:",
  errorTraining: "error on training text, smoothed",
  errorHeldBack: "error on held-back text",
  versionBest: "{parent}'s best, held-back",
  earlierVersion: "the earlier version",
  bitsPerCharNote: "All three in bits per character, so versions with different word lists compare fairly. Lower is better.",
  coolDownDrop: "Error usually drops most in the cool-down, where the learning rate falls to near zero.",
  trainingText: "training text, smoothed",
  heldBack: "held-back text",
  uniformGuess: "uniform guess",
  wordFrequencies: "counting word frequencies",
  earlierBest: "earlier best",
  versionsBest: "{parent}'s best",
  noPointsYet: "The first points appear at the next checkpoint.",
  chatStillUses: "Chat still uses {chat}.",
  noTrainedVersion: "no trained version yet",
  compareChat: "Compare them in Chat",
  howStatusWork: "How the status and schedule work",
  statusOneOf: "Status is one of three, decided from the run:",
  statusLearning: "Learning",
  learningDesc: "(smoothed error fell by more than the step-to-step noise over the last window),",
  statusFlat: "Flat",
  flatDesc: "(it didn't),",
  statusMemorising: "Memorising",
  memorisingDesc:
    "(held-back error rising while training error falls). Repeated passes over the same text help less each time and make memorising likely after a few; the fix is more text, or stopping at the best checkpoint.",
  scheduleNote:
    'Schedule: warm-up, then a long steady stretch, then a cool-down where the learning rate falls to near zero. Checkpoints taken mid-steady look worse than they are; "Stop after a short cool-down" and "Make a checkpoint now" (in Queue and schedule) cool down a copy first.',
  lrAutoNote:
    'Learning rate "auto": the earlier version\'s tuned rate, rescaled for the new width so it doesn\'t need retuning (μP). Change it only in Advanced.',
  gateHow:
    'How Chat decides: both versions are scored on the very same held-back passages, and the check asks whether the new version would still come out ahead if those passages had been drawn differently (it re-draws them 1,000 times). If it would nearly always, Chat switches. If not, it says "not sure yet" and keeps the earlier version: a small gap on a limited amount of text can be luck. It needs at least 20 held-back documents to judge. It looks once, when the run ends, because checking at every checkpoint would give luck many chances to win.',
  whatWritesNow: "What it writes now · same prompt every {every} steps",
  whatWritesNowPlain: "What it writes now · the same prompt each time",
  alwaysPicks: "It always picks the likeliest next piece, so only training changes the text.",
  step0: "step 0",
  identicalTo: "identical to {parent}: the new parts start at zero",
  randomStart: "random weights: it starts from scratch",
  nowStep: "now · step {step}",
  noSampleYet: "the first sample appears at the next checkpoint",
} as const;

export const STATUS = {
  learning: { label: "Learning", tone: "text-sky-400" },
  flat: { label: "Flat", tone: "text-muted-foreground" },
  memorising: { label: "Memorising", tone: "text-amber-400" },
} as const;

export const SWITCH: Record<string, string> = {
  held_back_inside:
    "{Version} is not ahead of {parent} on the held-back text yet. Chat checks once, when the run ends, and switches only if {version} is clearly better, or when you pick {version} yourself.",
  held_back_below:
    "{Version} is ahead of {parent} on the held-back text so far. That is only a first look: when the run ends, Chat checks both versions on the same held-back passages and switches only if the lead is too big to be luck, or when you pick {version} yourself.",
  no_band_yet:
    "Chat decides when the run ends. It scores {version} and the earlier best on the same held-back passages and switches only if the gain is too big to be luck, or when you pick {version} yourself.",
  gate_better:
    "{Version} beat {parent} on the same held-back passages by {gap} bits per character, and the gap is too big to be luck. Chat uses {version}.",
  gate_not_sure:
    "Not sure yet. {Version} is {gap} bits per character lower than {parent} on the same held-back passages, but a gap this small can still come from which passages were held back, so Chat keeps using {parent}.",
  gate_not_better:
    "{Version} is not better than {parent} on the same held-back passages ({gap} bits per character higher), so Chat keeps using {parent}.",
  gate_few_docs:
    "Not sure: only {docs} held-back documents; at least {need} are needed to tell a real gain from luck. Chat keeps using {parent}.",
};

export const GATE_SEEDS =
  "A different random start can also change the result. If you have run Compare with several seeds, its spread shows how much: a gap smaller than that spread is not a real gain.";

/** Fill {name} slots. */
export function fill(text: string, values: Record<string, string | number | null | undefined>): string {
  return text.replace(/\{(\w+)\}/g, (_m, key: string) => String(values[key] ?? ""));
}
