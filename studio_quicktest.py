"""Quick test: two short runs (current design vs a draft) on the same tiny data, order and wall-clock. Weights are discarded."""
from __future__ import annotations

import random
import tempfile
import threading
from pathlib import Path

import studio_eval
from studio_experiments import scaled_spec, shrunk, steps_for_seconds, train_one
from studio_model import LocalModel, Tokenizer, load_checkpoint
from studio_spec import config_from_spec

STORIES = [
    "Mira found a small blue boat on the beach. She pushed it into the water and it floated away. The next day it came back with a shell inside.",
    "A fox lived near the old mill. Every night it listened to the wheel turn. One night the wheel stopped and the fox went to see why.",
    "Tom planted a seed in a cup. He gave it water every morning. In a week a green shoot pushed up through the soil.",
    "The train was late again. Ana read her book while she waited. When the train came, she did not want to get on.",
    "An old clock hung in the hall. It was always five minutes slow. Nobody minded, because it ticked so kindly.",
    "Two crows argued about a shiny button. One said it was a star, the other said it was a coin. The button rolled away and ended the argument.",
    "Lena baked bread on Sundays. The whole street smelled of it. Even the dog sat by the gate and waited for a crust.",
    "A cloud got caught on the top of a hill. It stayed all afternoon and rained on one single field. The farmer called it his own little sky.",
    "Sam lost a glove in the snow. In spring he found it with a nest inside. He left the glove where it was.",
    "The lighthouse keeper counted ships. Some nights there were none. On those nights he counted stars instead.",
    "Rosa taught her parrot to say good morning. The parrot said it all day, and sometimes all night. Rosa slept very well anyway.",
    "A river ran past three villages. The first used it for washing, the second for fishing, the third just to watch it go by.",
]
MIN_GAP = .05  # bits per character; one short run each, so smaller gaps never count
CHOICES = {'stories': 'Short stories', 'slice': 'A slice of your data', 'problems': 'Practice problems'}


def tiny_dataset(name='stories', services=None, seed=0):
    """Plain-text examples for a quick test: the built-in stories, ~1% of the person's text, or generated problems."""
    if name == 'problems':
        import studio_rules
        rng, rows = random.Random(seed), []
        for rule in ('copy', 'reverse', 'sort', 'count'):
            rows += studio_rules.generate(rule, rng, 12)
        return [{'text': f"{r['prompt']} {r.get('answer', '')}".strip()} for r in rows]
    if name == 'slice' and hasattr(services, 'examples'):
        try:
            texts = [e['text'] for e in services.examples({'kind': 'text'}) if e.get('text')]
        except Exception:  # noqa: BLE001 - no usable workspace: fall back to the stories
            texts = []
        if texts:
            return [{'text': t} for t in texts[:max(12, len(texts) // 100)]]
    return [{'text': t} for t in STORIES]


def verdict(current_bpc, draft_bpc, threshold):
    """lower / about the same / higher: the draft's error against the current design's, with a clear-gap rule."""
    delta = draft_bpc - current_bpc
    return 'lower' if delta < -threshold else 'higher' if delta > threshold else 'about the same'


def _train(label, spec, examples, held, seconds, shrink, directory, emit, stop, extra):
    spec = shrunk(spec) if shrink else spec
    steps = steps_for_seconds(spec, examples, seconds, extra)
    out = train_one(scaled_spec(spec, steps), steps, 42, examples, held, directory, lambda e: emit(dict(e, type='quicktest', which=label)),
                    stop, None, extra, wall_seconds=seconds * 1.5)
    return spec, out


def run(current_spec, draft_spec, dataset='stories', minutes=1, shrink=False, emit=None, stop=None, config=None, services=None):
    """Train both designs for `minutes` each. Returns the early-signal verdict, curves, checks and samples.

    dataset is a name (stories | slice | problems) or a list of {'text'} examples. shrink = 1/4 width with muP.
    Progress: {type: 'quicktest', which: 'current' | 'draft', step, total, bpc}.
    """
    emit = emit or (lambda event: None)
    stop = stop or threading.Event()
    examples = tiny_dataset(dataset, services) if isinstance(dataset, str) else [e if isinstance(e, dict) else {'text': e} for e in dataset]
    train_docs, held = studio_eval.heldout_split([e['text'] for e in examples], .1)
    held = held or train_docs
    data = [{'text': t} for t in train_docs]
    seconds = max(1., minutes * 60.)
    result = {'minutes': minutes, 'shrink': bool(shrink), 'dataset': dataset if isinstance(dataset, str) else 'custom'}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        runs, specs = {}, {}
        for label, spec in (('current', current_spec), ('draft', draft_spec)):
            if stop.is_set():
                return dict(result, stopped=True)
            specs[label], runs[label] = _train(label, spec, data, held, seconds, shrink, tmp / label, emit, stop, config)
            if runs[label] is None:
                return dict(result, stopped=True)
        bands = [studio_eval.band_from_scores(studio_eval.score_docs(Path(r['directory']) / 'last.npz', held)) for r in runs.values()]
        threshold = max(MIN_GAP, *(hi - lo for lo, hi in bands))
        cur, dr = runs['current']['bpc'], runs['draft']['bpc']
        tok = Tokenizer.restore(load_checkpoint(Path(runs['draft']['directory']) / 'last.npz')['tokenizer'])
        refs = {'uniform': studio_eval.uniform_bpc(tok, held), 'unigram': studio_eval.unigram_bpc(tok, train_docs, held)}
        memorised = studio_eval.memorise_one_batch(dict(config_from_spec(specs['draft']), **(config or {})), train_docs[:4])
        checks = studio_eval.sanity_checks(runs['draft']['directory'], memorised=memorised)
        prompt = held[0][:24]
        for label in runs:
            model = LocalModel(Path(runs[label]['directory']) / 'last.npz', device='cpu')
            runs[label]['sample'] = prompt + model.generate([{'role': 'user', 'content': prompt}], max_tokens=60)
        result.update(verdict=verdict(cur, dr, threshold), delta=dr - cur, threshold=threshold, references=refs, checks=checks,
                      learned_context=dr < refs['unigram'],
                      **{k: {f: v for f, v in r.items() if f != 'directory'} for k, r in runs.items()})
    emit({'type': 'quicktest_done', 'results': result})
    return result
