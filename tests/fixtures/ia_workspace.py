"""Fixture workspace for testing IA pages: minimal real training on committed sample data."""
import json
import random
import time
from pathlib import Path

from studio_core import Workspace, normalize, parse_file
from studio_model import DEFAULT_CONFIG, train
from studio_data_kinds import add_problems
from studio_rules import generate as generate_problems


SAMPLE_PATH = Path(__file__).resolve().parent / "sample.jsonl"


def build(directory):
    """Create a fixture workspace with sample data and two trained versions.

    Returns the workspace. Re-entrant: if the workspace already exists and is complete,
    returns it without retraining.
    """
    directory = Path(directory)
    data_dir = directory / "data"

    # Check if already built (idempotence)
    if (data_dir / "workspace.json").exists() and (data_dir / "versions.json").exists():
        return Workspace(directory)

    # Create or load workspace
    ws = Workspace(directory)

    # Load sample data
    examples = [e for row in parse_file(SAMPLE_PATH) if (e := normalize(row))]
    if not examples:
        raise ValueError("No examples in sample.jsonl")

    # Create text dataset with first 50 examples
    text_dataset_id = ws.create_dataset("text", description="Sample math problems")
    ws.add(text_dataset_id, examples[:50], source="import")

    # Create 20 conversations (simple prompt/answer pairs)
    conversations_dataset_id = ws.create_dataset("conversations", description="Simple conversations")
    conversations = _generate_conversations(20, random.Random(42))
    ws.add(conversations_dataset_id, conversations, source="import")

    # Create 10 checkable problems using studio_rules
    problems_dataset_id = ws.create_dataset("problems", description="Checkable problems")
    problems = _generate_problems(10, random.Random(42))
    add_problems(ws, problems_dataset_id, problems)  # keeps the rule and params on each example

    # Prepare held-back documents (5 examples from the data)
    heldout = examples[50:55]
    heldout_list = [
        {"id": e.get("id", f"heldout_{i}"), "user": e.get("user", ""), "assistant": e.get("assistant", "")}
        for i, e in enumerate(heldout)
    ]
    (data_dir / "heldout.json").write_text(json.dumps(heldout_list, indent=2), encoding="utf-8")

    # Train two versions
    _train_versions(directory, examples[:50])

    # Write queue.json with defaults from BUILD.md S2
    queue_state = {
        "now": None,
        "up_next": [],
        "plan": {
            "hours": "any",
            "memory": "half",
            "idle_only": False,
            "pause_on_battery": False,
            "keep_awake": False,
            "chat_follows_best": False
        }
    }
    (data_dir / "queue.json").write_text(json.dumps(queue_state, indent=2), encoding="utf-8")

    return ws


def _generate_conversations(count, rng):
    """Generate simple prompt/answer conversation pairs."""
    topics = [
        ("What is machine learning?", "Machine learning is a subset of AI that learns patterns from data."),
        ("How do transformers work?", "Transformers use attention mechanisms to process sequences in parallel."),
        ("What is a token?", "A token is a unit of text that a model processes, like a word or subword."),
        ("Explain backpropagation.", "Backpropagation is an algorithm that computes gradients for neural networks."),
        ("What are embeddings?", "Embeddings are continuous vector representations of discrete items like words."),
        ("How does attention work?", "Attention computes weighted sums of values based on query-key similarities."),
        ("What is overfitting?", "Overfitting occurs when a model learns the training data too well, failing on new data."),
        ("Explain gradient descent.", "Gradient descent iteratively updates parameters in the direction of steepest descent."),
        ("What is a loss function?", "A loss function measures the difference between predictions and actual values."),
        ("How does batching help?", "Batching processes multiple examples together, improving computational efficiency."),
    ]

    conversations = []
    for i in range(count):
        user_msg, assistant_msg = topics[i % len(topics)]
        conversations.append({
            "id": f"conv_{i}",
            "user": user_msg,
            "assistant": assistant_msg,
            "intent": "qa"
        })
    return conversations


def _generate_problems(count, rng):
    """Generate checkable problems from studio_rules."""
    problems = []
    rules = ["copy", "reverse", "sort", "count", "small_addition", "max_length", "ends_cleanly"]

    for i in range(count):
        rule = rules[i % len(rules)]
        generated = generate_problems(rule, rng, 1)
        problems.append(generated[0])

    return problems


def _train_versions(directory, examples):
    """Train two versions: v1 default + small, v2 v1 with more layers via warm_start."""
    directory = Path(directory)
    runs_dir = directory / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Version 1: lightweight config
    cfg1 = {
        **DEFAULT_CONFIG,
        "width": 16,
        "layers": 1,
        "heads": 2,
        "context": 32,
        "steps": 30,
        "batch_size": 4,
        "learning_rate": 0.01,
        "warmup": 3,
        "eval_every": 5,
        "dropout": 0.0,
        "device": "cpu",
        "seed": 42,
    }

    v1_dir = runs_dir / "v1"
    start = time.time()
    record1 = train(cfg1, examples, v1_dir, emit=_null_emit, stop=None, pause=None)
    v1_time = time.time() - start

    if record1["status"] != "completed":
        raise RuntimeError(f"Version 1 training failed: {record1}")

    # Version 2: warm-start from v1 with more layers
    cfg2 = {**cfg1, "layers": 3, "steps": 10}
    v2_dir = runs_dir / "v2"

    # init_from transplants every matching tensor from version 1 (the engine's warm start); the two new blocks start fresh
    start = time.time()
    record2 = train(cfg2, examples, v2_dir, emit=_null_emit, stop=None, pause=None, init_from=v1_dir / "last.npz")
    v2_time = time.time() - start

    if record2["status"] != "completed":
        raise RuntimeError(f"Version 2 training failed: {record2}")

    # Write versions.json
    versions = [
        {
            "number": 1,
            "change": "initial version",
            "parent": None,
            "spec": cfg1,
            "created_at": json.dumps(time.time()),
            "runs": [{"dir": str(v1_dir), "steps": cfg1["steps"], "duration_s": v1_time}],
            "best_checkpoint": str(v1_dir / "best.npz"),
            "status": "trained",
            "scorecard": {},
            "verdict": "baseline"
        },
        {
            "number": 2,
            "change": "layers 1 → 3",
            "parent": 1,
            "spec": cfg2,
            "created_at": json.dumps(time.time()),
            "runs": [{"dir": str(v2_dir), "steps": cfg2["steps"], "duration_s": v2_time}],
            "best_checkpoint": str(v2_dir / "best.npz"),
            "status": "trained",
            "scorecard": {},
            "verdict": "iteration"
        }
    ]

    (directory / "data" / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")


def _config_to_spec(config):
    """Convert a config dict to a minimal spec for warm_start."""
    # This is a stub; in real usage, studio_versions or studio_spec would provide this.
    return {
        "blocks": [{"type": "transformer"} for _ in range(config["layers"])],
        "embedding": {
            "tokenizer": config.get("tokenizer", "byte"),
            "width": config["width"],
            "context": config["context"],
        }
    }


def _null_emit(event):
    """No-op emit callback."""
    pass
