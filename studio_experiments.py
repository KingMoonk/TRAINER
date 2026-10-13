"""Repeatable comparisons: explicit matching adjustments, paired seeds and worker events."""
from __future__ import annotations

import contextlib
import copy
import json
import math
import random
import sys
import threading
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from studio_core import atomic_json
from studio_eval import bits_per_char, heldout_split
from studio_estimate import estimate
from studio_spec import config_from_spec, validate_spec
from studio_validate import errors

MODES = ('params', 'compute', 'tokens', 'wall-clock')
SEEDS = (7, 42, 123, 7919, 31337)
LR_FACTORS = (.5, 1., 2.)
DATA_SEED = 1234  # one data order for every run


@dataclass
class ExperimentSpec:
    base: dict
    variants: list = field(default_factory=list)
    seeds: list | int = field(default_factory=lambda: [7, 42, 123])  # an int N means the first N of SEEDS
    mode: str = 'tokens'
    token_budget: int = 10000
    wall_seconds: float = 60.
    steps: int = 100
    question: str = ''
    lr_retune: bool = True
    guess: str = ''
    names: list = field(default_factory=list)  # optional display names: base first, then the variants
    schema: int = 1

    def seed_list(self):
        return list(SEEDS[:self.seeds]) if isinstance(self.seeds, int) and not isinstance(self.seeds, bool) and 1 <= self.seeds <= len(SEEDS) else self.seeds

    def normalized(self):
        if self.schema != 1 or self.mode not in MODES:
            raise ValueError('Unsupported experiment schema or matching mode.')
        seeds = self.seed_list()
        if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s <= 2147483647 for s in seeds):
            raise ValueError('Choose distinct integer seeds between 0 and 2147483647.')
        if self.token_budget < 1 or self.wall_seconds <= 0 or not 1 <= self.steps <= 1000000:
            raise ValueError('Experiment budgets must be positive and steps at most 1,000,000.')
        result = copy.deepcopy(self)
        result.seeds = seeds
        result.base = validate_spec(self.base)
        result.variants = [validate_spec(value) for value in self.variants]
        for value in result.variants:
            if value['tokenizer'] != result.base['tokenizer']:
                raise ValueError('Comparisons must share the same tokenizer and token accounting.')
        return result

    def save(self, path):
        atomic_json(path, asdict(self.normalized()))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())).normalized()


def match(base, variant, mode='params', vocab=None, tolerance=.02):
    """Match estimated parameters/compute by a single shared channel-width adjustment.

    Return an explicit impossible result when this knob cannot achieve tolerance.
    Token and wall-clock matching leave architecture unchanged and use runtime budgets.
    """
    if mode not in MODES:
        raise ValueError('Unknown hold-equal mode.')
    base, variant = validate_spec(base), validate_spec(variant)
    if mode in ('tokens', 'wall-clock'):
        return {'spec': variant, 'adjustments': [], 'matched': True, 'relative_error': 0., 'mode': mode}
    key = 'total' if mode == 'params' else 'train_flops_per_token'
    target = estimate(base, vocab)[key]
    actual = estimate(variant, vocab)[key]
    best, best_actual = variant, actual
    if target <= 0:
        return {'spec': best, 'adjustments': [], 'matched': actual == target, 'relative_error': 0. if actual == target else float('inf'), 'mode': mode}
    if abs(actual - target) / target > tolerance and 'd_ff' in variant['block']['channel']['config']:
        # Keep explicitly overridden blocks untouched. Search the shared default width only.
        low, high = 1, 8192
        while low <= high:
            width = (low + high) // 2
            candidate = copy.deepcopy(variant)
            candidate['block']['channel']['config']['d_ff'] = width
            candidate = validate_spec(candidate)
            value = estimate(candidate, vocab)[key]
            if abs(value - target) < abs(best_actual - target) and not errors(candidate, vocab):
                best, best_actual = candidate, value
            if value < target:
                low = width + 1
            else:
                high = width - 1
    changes = []
    if best != variant:
        changes.append({'field': 'block.channel.config.d_ff', 'before': variant['block']['channel']['config']['d_ff'],
                        'after': best['block']['channel']['config']['d_ff']})
    relative = abs(best_actual - target) / target
    return {'spec': best, 'adjustments': changes, 'matched': relative <= tolerance, 'relative_error': relative,
            'target': target, 'actual': best_actual, 'mode': mode,
            'reason': '' if relative <= tolerance else 'Shared channel width cannot match this budget within tolerance.'}


def seed_stats(values, base=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Seed statistics require finite losses.')
    result = {'mean': float(values.mean()), 'min': float(values.min()), 'max': float(values.max()),
              'std': float(values.std(ddof=1)) if len(values) > 1 else None, 'count': len(values),
              'interpretation': 'Descriptive seed spread; not a significance test.'}
    if base is not None:
        base = np.asarray(base, dtype=float)
        if base.shape != values.shape or not np.isfinite(base).all():
            raise ValueError('Pair each variant with the same completed base seeds.')
        delta = values - base
        result['delta'] = float(delta.mean())
        result['within_noise'] = None if len(values) < 2 else abs(result['delta']) <= max(float(np.ptp(base)), float(np.ptp(values)))
    return result


def run(experiment, examples, directory, emit=None, stop=None, config=None):
    from studio_model import train, validate_config
    experiment = experiment.normalized()
    emit = emit or (lambda event: None)
    stop = stop or threading.Event()
    directory = Path(directory)
    if (directory / 'experiment.json').exists():
        raise ValueError('Choose a new experiment directory.')
    directory.mkdir(parents=True, exist_ok=True)
    experiment.save(directory / 'experiment.json')
    runs = []
    for index, spec in enumerate([experiment.base] + experiment.variants):
        matched = match(experiment.base, spec, experiment.mode)
        if not matched['matched']:
            raise ValueError(f"Variant {index}: {matched['reason']}")
        for seed in experiment.seeds:
            if stop.is_set():
                break
            cfg = dict(config or {})
            cfg.update(config_from_spec(matched['spec']))
            cfg.update(seed=seed, steps=experiment.steps)
            cfg = validate_config(cfg)
            path = directory / f'variant_{index}_seed_{seed}'
            def event(value):
                emit(dict(value, variant=index, seed=seed))
            emit({'type': 'experiment_run', 'variant': index, 'seed': seed, 'adjustments': matched['adjustments']})
            record = train(cfg, examples, path, emit=event, stop=stop,
                           token_budget=experiment.token_budget if experiment.mode == 'tokens' else None,
                           wall_seconds=experiment.wall_seconds if experiment.mode == 'wall-clock' else None)
            runs.append({'variant': index, 'seed': seed, 'directory': str(path), 'record': record,
                         'adjustments': matched['adjustments']})
            atomic_json(directory / 'results.json', runs)
        if stop.is_set():
            break
    emit({'type': 'experiment_done', 'runs': runs, 'cancelled': stop.is_set()})
    return runs


def start(experiment, examples, directory, queue, stop=None, config=None):
    def worker():
        try:
            run(experiment, examples, directory, emit=queue.put, stop=stop, config=config)
        except Exception as error:
            queue.put({'type': 'experiment_error', 'error': str(error)})
    thread = threading.Thread(target=worker, daemon=True, name='studio-experiment')
    thread.start()
    return thread


# ---- planned comparisons (the "compare" queue job) -------------------------------------------------------------
LN2 = math.log(2)
_PATCH = threading.Lock()


def lr_of(spec):
    return spec['train']['optimizer']['config']['lr']


def scaled_spec(spec, steps, lr=None):
    """Copy of spec on the WSD schedule scaled to `steps`: warm-up 5% of the run, decay fraction preserved at 20%."""
    spec = copy.deepcopy(spec)
    spec['train']['schedule'] = {'id': 'wsd', 'config': {'warmup': max(1, round(.05 * steps)), 'stable_fraction': .8}}
    if lr is not None:
        spec['train']['optimizer']['config']['lr'] = lr
    return validate_spec(spec)


def shrunk(spec, factor=4):
    """The spec at 1/factor width. Learning rate carries over by muP with the full width as the base width."""
    spec = copy.deepcopy(spec)
    full, heads = spec['dims']['d_model'], max(1, spec['block']['mixer']['config'].get('heads', 1))
    spec['dims']['d_model'] = -(-max(16, full // factor) // heads) * heads  # engine minimum 16, a multiple of the heads
    ff = spec['block']['channel']['config']
    if ff.get('d_ff'):
        ff['d_ff'] = max(1, ff['d_ff'] // factor)
    family = 'sgd' if spec['train']['optimizer']['id'] == 'sgd' else 'adam'
    spec['train']['init'] = {'id': 'mup', 'config': {'base_width': full, 'optimizer_family': family}}
    return validate_spec(spec)


@contextlib.contextmanager
def data_order(seed=DATA_SEED):
    """Pin the batch sampler to one seed while cfg['seed'] varies the initial weights.

    ponytail: studio_model.train seeds both from cfg['seed'] and uses `random` only for the sampler, so the module
    attribute is swapped for the call. Replace with a data_seed config key once the engine has one.
    """
    import studio_model
    with _PATCH:
        saved = studio_model.random
        studio_model.random = types.SimpleNamespace(Random=lambda _seed: random.Random(seed))
        try:
            yield
        finally:
            studio_model.random = saved


def _peak_mb():
    if sys.platform == "win32":  # no resource module; ponytail: current working set, not the peak
        from studio_passive import process_memory
        return process_memory() / 2 ** 20
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KiB on Linux
    return peak / 2 ** 20 if sys.platform == 'darwin' else peak / 1024


def best_lr(points):
    """[(lr, final bpc)] -> the lr with the lowest finite bpc (the base rate when none is finite)."""
    good = [(b, lr) for lr, b in points if math.isfinite(b)]
    return min(good)[1] if good else points[1][0]


def spec_changes(base, variant, limit=6):
    """Raw knob changes as 'path: before → after' strings (depth 3: the exact values behind the plain-words change)."""
    def flat(value, path=''):
        if isinstance(value, dict):
            return {k: v for key, item in value.items() for k, v in flat(item, f'{path}.{key}' if path else key).items()}
        return {path: value}
    a, b = flat(validate_spec(base)), flat(validate_spec(variant))
    out = [f'{k}: {a.get(k, "none")} → {b.get(k, "none")}' for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    return out[:limit] + ([f'… {len(out) - limit} more'] if len(out) > limit else [])


def variant_outcome(row):
    """What happened to a variant against the base: better | about the same | worse, or None when no verdict is possible.

    A difference counts only when it is larger than the spread across seeds, so one seed (no spread) gives no verdict.
    """
    if not row or row.get('delta') is None or row.get('inside_spread') is None:
        return None
    return 'about the same' if row['inside_spread'] else 'better' if row['delta'] < 0 else 'worse'


def load_results(run_dir, job_id):
    """The saved results.json of a finished compare job, or None."""
    path = Path(run_dir) / f'compare_{job_id}' / 'results.json'
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def verdict_inside(delta, spread):
    return None if delta is None or spread is None else abs(delta) <= spread


def summarize(runs):
    """runs: [{variant, seed, bpc, curve, pieces_per_s, peak_mb, lr}] -> one row per variant, base = variant 0.

    spread = max-min across seeds; vs base = paired mean delta; inside_spread when |delta| <= the larger spread of the
    two (seed_stats). A single seed has no spread, so inside_spread is None.
    """
    by = {}
    for run in sorted(runs, key=lambda r: (r['variant'], r['seed'])):
        by.setdefault(run['variant'], []).append(run)
    base = [r['bpc'] for r in by.get(0, [])]
    rows = []
    for variant, items in sorted(by.items()):
        values = [r['bpc'] for r in items]
        stats = seed_stats(values, base if variant and len(base) == len(values) else None)
        grid = [p[0] for p in items[0]['curve']]
        mean = np.mean([np.interp(grid, [p[0] for p in r['curve']], [p[1] for p in r['curve']]) for r in items], axis=0) if grid else []
        rows.append({'variant': variant, 'seeds': [r['seed'] for r in items], 'bpc': values, 'mean': stats['mean'],
                     'min': stats['min'], 'max': stats['max'], 'spread': stats['max'] - stats['min'],
                     'delta': stats.get('delta'), 'inside_spread': stats.get('within_noise'), 'lr': items[0].get('lr'),
                     'pieces_per_s': float(np.mean([r['pieces_per_s'] for r in items])),
                     'peak_mb': max(r['peak_mb'] for r in items), 'curve': [[float(x), float(y)] for x, y in zip(grid, mean)]})
    return rows


def plan(experiment):
    """Expand to the run list: base + variants, each run per seed, plus a 3-point LR sweep per non-base variant.

    steps is None for wall-clock mode (resolved from a timing probe when the job runs). The base keeps its own rate
    and is not swept, so 3 variants + base x 3 seeds = 12 runs + 9 sweeps.
    """
    e = experiment.normalized()
    specs = [e.base] + e.variants
    base_flops = estimate(e.base)['train_flops_per_token']
    rows = []
    for index, spec in enumerate(specs):
        matched = match(e.base, spec, e.mode)
        steps = None if e.mode == 'wall-clock' else e.steps
        if e.mode == 'compute':
            steps = max(1, round(e.steps * base_flops / max(1, estimate(matched['spec'])['train_flops_per_token'])))
        rows.append({'variant': index, 'name': (e.names[index] if index < len(e.names) else 'Base' if not index else f'Variant {index}'),
                     'spec': matched['spec'], 'adjustments': matched['adjustments'], 'matched': matched['matched'],
                     'reason': matched.get('reason', ''), 'steps': steps, 'parameters': estimate(matched['spec'])['total'],
                     'relative_error': matched['relative_error']})
    sweeps = [{'variant': r['variant'], 'factor': f, 'steps': None if r['steps'] is None else max(5, r['steps'] // 5)}
              for r in rows[1:] for f in LR_FACTORS] if e.lr_retune else []
    runs = [{'variant': r['variant'], 'seed': s, 'steps': r['steps']} for r in rows for s in e.seeds]
    return {'variants': rows, 'runs': runs, 'sweeps': sweeps, 'n_runs': len(runs), 'n_sweeps': len(sweeps), 'seeds': e.seeds}


def _config(spec, steps, seed, extra=None):
    cfg = dict(extra or {})
    cfg.update(config_from_spec(spec))
    cfg.update(seed=seed, steps=steps, validation=0., objective='all', eval_every=max(1, steps))
    return cfg


def steps_for_seconds(spec, examples, seconds, extra=None, probe_steps=8):
    """Steps that fit `seconds` on this machine, from a short timed probe (at least 5)."""
    import tempfile

    from studio_model import train
    with tempfile.TemporaryDirectory() as tmp, data_order():
        train(_config(spec, probe_steps, 1, extra), examples, Path(tmp) / 'probe')
        rows = [json.loads(line) for line in (Path(tmp) / 'probe' / 'metrics.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r.get('step', 0) >= 1 and 'elapsed_seconds' in r]
    per_step = (rows[-1]['elapsed_seconds'] - rows[0]['elapsed_seconds']) / max(1, rows[-1]['step'] - rows[0]['step'])
    return max(5, int(seconds / max(per_step, 1e-4)))


def train_one(spec, steps, seed, examples, held, directory, emit, stop, pause=None, extra=None, wall_seconds=None, tag=None):
    """One run on the pinned data order. Returns {bpc, curve, pieces_per_s, peak_mb, steps, directory} or None if stopped."""
    from studio_model import Tokenizer, load_checkpoint, train
    directory = Path(directory)
    before = _peak_mb()

    def progress(event):
        if event.get('type') == 'metric' and event.get('step') and event.get('loss') is not None:
            emit(dict(tag or {}, type='compare', step=event['step'], total=steps, bpc=event['loss'] / LN2))
    try:
        with data_order():
            record = train(_config(spec, steps, seed, extra), examples, directory, emit=progress, stop=stop, pause=pause,
                           wall_seconds=wall_seconds)
    except InterruptedError:
        return None
    if stop is not None and stop.is_set():
        return None
    bpc = bits_per_char(directory / 'last.npz', held)
    tokenizer = Tokenizer.restore(load_checkpoint(directory / 'last.npz')['tokenizer'])
    chars = sum(len(e['text']) for e in examples)
    pieces = sum(len(tokenizer.encode(e['text'])) for e in examples)
    scale = pieces / max(1, chars) / LN2  # nats per piece -> bits per character
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    curve = [[r['step'], r['loss'] * scale] for r in rows if r.get('loss') is not None]
    return {'bpc': bpc, 'curve': curve, 'steps': record['step'], 'directory': str(directory),
            'pieces_per_s': record['trained_tokens'] / max(.001, record['duration_seconds']),
            'peak_mb': max(0., _peak_mb() - before)}


def execute(experiment, examples, directory, emit=None, stop=None, pause=None, config=None):
    """Run a planned comparison one run after another; writes results.json and returns the result dict.

    Sweeps come first (variants only, first seed, a fifth of the steps, rate x0.5/x1/x2 of the base rate); each
    variant then trains every seed at its best rate. Progress: {type: 'compare', variant, seed, step, bpc, total}.
    """
    e = experiment.normalized()
    emit = emit or (lambda event: None)
    stop = stop or threading.Event()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    e.save(directory / 'experiment.json')
    pl = plan(e)
    bad = [r for r in pl['variants'] if not r['matched']]
    if bad:
        raise ValueError(f"Variant {bad[0]['variant']}: {bad[0]['reason']}")
    texts = [x['text'] for x in examples if x.get('text')]
    train_docs, held = heldout_split(texts, .1)
    held = held or train_docs
    data = [{'text': t} for t in train_docs]
    base_lr, lrs, sweep_out, runs_out = lr_of(e.base), {}, [], []
    steps = {}
    for row in pl['variants']:
        steps[row['variant']] = row['steps'] or steps_for_seconds(row['spec'], data, e.wall_seconds, config)
    for row in pl['variants']:
        index = row['variant']
        lrs[index] = base_lr
        if index == 0 or not e.lr_retune:
            continue
        points = []
        for factor in LR_FACTORS:
            if stop.is_set():
                break
            n = max(5, steps[index] // 5)
            out = train_one(scaled_spec(row['spec'], n, base_lr * factor), n, e.seeds[0], data, held,
                            directory / f'sweep_{index}_x{factor}', emit, stop, pause, config,
                            tag={'variant': index, 'seed': e.seeds[0], 'phase': 'sweep'})
            if out:
                points.append((base_lr * factor, out['bpc']))
                sweep_out.append({'variant': index, 'factor': factor, 'lr': base_lr * factor, 'bpc': out['bpc']})
        if len(points) == len(LR_FACTORS):
            lrs[index] = best_lr(points)
    for row in pl['variants']:
        for seed in e.seeds:
            if stop.is_set():
                break
            index = row['variant']
            emit({'type': 'compare_run', 'variant': index, 'seed': seed})
            out = train_one(scaled_spec(row['spec'], steps[index], lrs[index]), steps[index], seed, data, held,
                            directory / f'variant_{index}_seed_{seed}', emit, stop, pause, config,
                            tag={'variant': index, 'seed': seed, 'phase': 'run'})
            if out:
                runs_out.append(dict(out, variant=index, seed=seed, lr=lrs[index]))
    from studio_eval import unigram_bpc, uniform_bpc
    refs = {}
    if runs_out:
        from studio_model import Tokenizer, load_checkpoint
        tok = Tokenizer.restore(load_checkpoint(Path(runs_out[0]['directory']) / 'last.npz')['tokenizer'])
        refs = {'uniform': uniform_bpc(tok, held), 'unigram': unigram_bpc(tok, train_docs, held)}
    result = {'variants': summarize(runs_out) if runs_out else [], 'sweeps': sweep_out, 'references': refs,
              'cancelled': stop.is_set(), 'lr_retuned': e.lr_retune,
              'runs': [{k: v for k, v in r.items() if k != 'curve'} for r in runs_out]}
    atomic_json(directory / 'results.json', result)
    emit({'type': 'compare_done', 'results': result})
    return result
