export type LossResult = {
  tokens: string[]; pos: number; probabilities: number[]; target: number; pieces: string[]; grad_error?: number | null;
};

export function lossView(data: LossResult) {
  const { probabilities: p, pieces, target } = data;
  if (!Number.isInteger(target) || target < 0 || target >= p.length || pieces.length !== p.length ||
      p.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
    throw new Error("The model returned an invalid probability distribution.");
  }
  const top = p.map((_, index) => index).sort((a, b) => p[b] - p[a]).slice(0, 5);
  const rows = [target, ...top.filter((index) => index !== target)].map((index) => ({
    index, piece: pieces[index], probability: p[index], target: index === target ? 1 : 0,
    gradient: p[index] - (index === target ? 1 : 0),
  }));
  const loss = -Math.log(Math.max(p[target], 1e-300));
  const characters = Math.max(1, Array.from(pieces[target]).length);
  return { rows, loss, characters, bits: loss / (characters * Math.LN2),
    remaining: Math.max(0, 1 - rows.reduce((sum, row) => sum + row.probability, 0)),
    others: p.length - rows.length, piece: pieces[target], probability: p[target] };
}

export const showPiece = (piece: string) => piece.replaceAll(" ", "·").replaceAll("\n", "↵").replaceAll("\t", "⇥");
export const probability = (value: number) => value < 0.001 ? value.toExponential(1) : value.toFixed(2);
