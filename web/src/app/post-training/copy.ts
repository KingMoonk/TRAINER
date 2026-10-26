// The strings this page uses.

export const T = {
  title: "Post-training",
  lead: "Four rounds, always in this order. Same design throughout; only the weights shift.",
  noVersions: "No versions yet. Build one in the Network designer, or start from a pretrained model below.",
  startsFrom: "Starts from",
  noVersion: "No trained version yet",
  atThisSize: "At this size the gain from each round is small. The shape of the pipeline is what you are learning.",
  loraLabel: "What it changes",
  loraOff: "Every weight",
  loraOn: "LoRA adapters",
  loraRankLabel: "Adapter size (rank)",
  loraDescOff: "Ordinary training: every weight in the model can change.",
  loraDescOn:
    "The model's own weights stay frozen. Only small add-on layers learn, so the round is lighter and the model drifts less from what it already knows. A bigger rank can learn more and costs more memory.",
  teacherLabel: "Teacher",
  previousBest: "Another trained version",
  thisModelCorrection: "This model, shown your correction as a hint",
  previousDesc:
    "{version}. Same word list and reading window as this model, and the earlier version with the lowest answer loss among those whose Round 1 passed.",
  noDefaultTeacher: "No earlier version has a passed Round 1 yet, so no teacher is preselected: pick one below, or run Round 1 on an earlier version first.",
  teacherChatPassed: "answer loss {loss}, Round 1 passed",
  teacherChatFailed: "answer loss {loss}, Round 1 did not pass",
  teacherNoChat: "Round 1 not run",
  noPrevious: "No other trained version shares this model's word list yet: train another first, or teach a correction in Chat.",
  correctionDesc:
    "From Chat's \"Teach a correction\": the same model is shown your correction as a hint and learns to answer as if it had it. This only works if the model can use a hint.",
  whichVersion: "Which version teaches",
  teacherNote:
    "The teacher must share this model's word list and reading window; the app checks and says so if it doesn't. Your local model (Ollama) can't teach here. It keeps two jobs: rewriting prompts and judging.",
  howFar: "How far it may move",
  leashNote: "Only pairs a person rated count here; the judge model's picks never do. Optional, and always last.",
  forEachPair:
    "For each pair you rated, the learning copy widens the gap between your pick and the other, compared with the frozen starting copy. Both can become less likely; the gap is what it is trained on.",
  fewThousand:
    "A few thousand conversations teach the chat format; more keep improving the answers. It trains about two passes over all your conversations, up to 2,000 steps, so a large mix takes a while. Only the answer part is scored. No frozen copy and no leash: this round is ordinary training on good examples.",
  rulesCheck:
    "Rules it can check: copy, reverse, sort, count, small addition, stay under a length, end cleanly, and the answer key an imported set came with (a final number or an option letter). A teacher's rubric can add a smaller second score, but an answer that fails a rule gets the minimum whatever the rubric says.",
  spreadSome: "{taught} of {groups} {groupsWord} had both a better and a worse answer; only those teach the model.",
  spreadNone: "None of the {groups} {groupsWord} had both a better and a worse answer, so nothing was learned.",
  spreadHint: "Only a group with both a better and a worse answer teaches the model. How many did is shown here as it trains.",
  notSurprised: "not surprised",
  verySurprised: "very surprised · pushed toward what the teacher would write",
  answerShaded: "Its answer, shaded by how surprised the teacher was by each word",
  chatMoves: "Chat moves to the new checkpoint only if no check below fails:",
  judgeNote: "shown for context; never decides alone",
  heldBackCheck: "Held-back text: error did not rise",
  conversationsCheck: "Held-back conversations: answer error fell",
  stopCheck: "Answers still stop: end-of-answer rate did not fall",
  gatePassed: "Passed",
  gateFailed: "Failed",
  gateUnmeasured: "Not measured",
  verdictFailed: "This round did not pass its checks.",
  verdictUnmeasured: "Nothing could be measured for this round, so it is not counted as a pass.",
  problemsCheck: "Checked problems: pass rate did not fall",
  measuredAtEnd: "measured when the round ends",
  judgeNone: "Judge: not run yet",
  willCount: "Will count {what}.",
  soFarCount: "So far: counting {what} (on training prompts; held-back check at the end).",
  soFar: "So far: {verdict} (on training prompts; held-back check at the end)",
  willRead: "Will read: \"{verdict}\"",
  resultKept: "Kept for comparison, not live. Chat stays on the previous checkpoint.",
  resultLive: "Live: Chat uses {checkpoint} from the next message.",
  chatKeeps: "Chat keeps {checkpoint} until this round passes its checks.",
  untilThen: "Until then, Chat keeps its current checkpoint.",
  overlapLeftOut: "{leftOut} of {checked} held-back {what} also appear in training and were left out.",
  moveFront: "Move to the front of the queue",
  addVotes: "Add votes round to the queue",
  add: "Add to the queue",
  addConversations: "Add conversations in Your data",
  makeProblems: "Make problems a rule can check",
  ratePairs: "Rate pairs in Chat",
} as const;

export function fill(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (_m, key: string) => String(values[key] ?? ""));
}

export type RoundSpec = {
  n: 1 | 2 | 3 | 4;
  name: string;
  kicker: string;
  method: string;
  lead: string;
  verdict: string;
  foot: string;
  counted: boolean;
  action: string;
  link?: { label: string; href: string };
};

export const ROUNDS: RoundSpec[] = [
  {
    n: 1,
    name: "Practice answers",
    kicker: "Round 1 · Practice answers",
    method: "supervised fine-tuning",
    lead: "Teaches the shape of an answer, not the skill. Pretraining taught it to continue text; here it studies prompts paired with good answers.",
    verdict: "Answers now look like answers.",
    foot: "Corrections you teach in Chat that need a whole new answer also land here.",
    counted: false,
    action: T.add,
    link: { label: T.addConversations, href: "/your-data" },
  },
  {
    n: 2,
    name: "Practice with checks",
    kicker: "Round 2 · Practice with checks",
    method: "reinforcement learning with rule-checked rewards",
    lead: "For each prompt it writes 6 answers. A rule scores each one. Answers above the group's average are made more likely; answers below it, less likely. A group where every answer scores the same teaches nothing.",
    verdict: "Answers that passed the check became more likely than ones that failed.",
    foot: "Prompts where all answers pass, or all fail, teach nothing and are skipped. Before it starts, the model answers a few of the problems; if it solves none, the round is not run, since every group would teach nothing. Each rule gets the room its answers need, up to 96 word pieces, so a step takes longer than it would with every answer cut at 32.",
    counted: true,
    action: T.add,
    link: { label: T.makeProblems, href: "/your-data/?kind=problems#make-extra-data" },
  },
  {
    n: 3,
    name: "Learn from a teacher",
    kicker: "Round 3 · Learn from a teacher",
    method: "distillation from a teacher, or from the same model shown a hint",
    lead: "It writes its own answer and a teacher scores every word. The teacher is another trained version, or this same model shown a hint, so it learns to answer as if it had the hint. A hint only helps if the model can use one, and tiny models often can't.",
    verdict: "how many words of its answers moved toward the teacher's",
    foot: "Also runs on its own after a night of training on new text: new text can erode how it answers, and a short lesson from the previous best version restores it without losing the new text.",
    counted: true,
    action: T.moveFront,
  },
  {
    n: 4,
    name: "Your votes",
    kicker: "Round 4 · Your votes",
    method: "direct preference optimisation (DPO)",
    lead: "Two answers to one prompt; you pick the better one. Training widens the gap between your pick and the other.",
    verdict: "on how many held-back pairs the gap between your pick and the other widened",
    foot: "1 in 5 rated pairs is held back to check the result.",
    counted: false,
    action: T.addVotes,
    link: { label: T.ratePairs, href: "/chat" },
  },
];

export const STATUS_TEXT: Record<string, string> = { done: "done", running: "running", queued: "in the queue", idle: "not started" };

export const ROW_KINDS = [
  { value: "conversations", label: "Conversations" },
  { value: "problems", label: "Problems" },
  { value: "pairs", label: "Rated pairs" },
  { value: "corrections", label: "Corrections" },
  { value: "heldout_docs", label: "Held-back text" },
  { value: "heldout_problems", label: "Held-back problems" },
] as const;
