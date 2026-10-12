"""Repeatable comparisons: explicit matching adjustments, paired seeds and worker events."""
from __future__ import annotations

import copy
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from studio_core import atomic_json
from studio_estimate import estimate
from studio_spec import config_from_spec, validate_spec
from studio_validate import errors

MODES = ('params', 'compute', 'tokens', 'wall-clock')


@dataclass
class ExperimentSpec:
    base: dict
    variants: list = field(default_factory=list)
    seeds: list = field(default_factory=lambda: [7, 42, 123])
    mode: str = 'tokens'
    token_budget: int = 10000
    wall_seconds: float = 60.
    steps: int = 100
    question: str = ''
    schema: int = 1

    def normalized(self):
        if self.schema != 1 or self.mode not in MODES:
            raise ValueError('Unsupported experiment schema or matching mode.')
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s <= 2147483647 for s in self.seeds):
            raise ValueError('Choose distinct integer seeds between 0 and 2147483647.')
        if self.token_budget < 1 or self.wall_seconds <= 0 or not 1 <= self.steps <= 1000000:
            raise ValueError('Experiment budgets must be positive and steps at most 1,000,000.')
        result = copy.deepcopy(self)
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
