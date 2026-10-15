"""Version-pinned public training corpora, verified downloads and disk-backed imports.

Only dataset files are fetched. This module never downloads model weights or calls
an inference service. Original source files, dataset cards and checksums remain
beside the converted JSONL so the training material can be inspected and attributed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import random
import re
import shutil
import threading
import time
import urllib.request
from pathlib import Path

from studio_core import ROOT, atomic_json, fingerprint, normalize, now, replace_file, truthy, uid

_log = logging.getLogger(__name__)


def _hf(repo, revision, filename):
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}?download=true"


# An entry may carry "sha256": the expected hash of the fetched source bytes (the range_bytes prefix when
# bounded). When present, a mismatching download is rejected; without it the receipt is trust-on-first-use.
CATALOG = [
    # --- stage "language": plain text for the first, next-character-prediction stage -------------
    {"id": "tinystories", "name": "TinyStories · simple stories", "domain": "Language foundation", "stage": "language", "default": True,
     "description": "The ideal first-stage corpus for tiny models: short, simple GPT-4-written children's stories. Bounded to the first ~64 MB of the V2 training file (whole stories only) as plain text.",
     "license": "CDLA-Sharing-1.0", "source_url": "https://huggingface.co/datasets/roneneldan/TinyStories",
     "size_hint": "64 MB download (bounded prefix)", "records_hint": "77,941 stories", "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
     "repo": "roneneldan/TinyStories", "filename": "TinyStoriesV2-GPT4-train.txt", "format": "txt_stories", "parser": "story",
     "range_bytes": 64_000_000, "split": "train", "synthetic": True},
    {"id": "tinystories_valid", "name": "TinyStories · validation stories", "domain": "Language foundation", "stage": "language", "default": False,
     "description": "The published V2 validation file of TinyStories, kept as plain-text held-out stories for measuring language-stage loss.",
     "license": "CDLA-Sharing-1.0", "source_url": "https://huggingface.co/datasets/roneneldan/TinyStories",
     "size_hint": "22 MB download", "records_hint": "27,630 stories", "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
     "repo": "roneneldan/TinyStories", "filename": "TinyStoriesV2-GPT4-valid.txt", "format": "txt_stories", "parser": "story",
     "split": "validation", "synthetic": True},
    {"id": "wikitext2", "name": "WikiText-2 · Wikipedia prose", "domain": "Language foundation", "stage": "language", "default": True,
     "description": "Verified-good Wikipedia articles (training split of WikiText-2, raw), split into paragraphs and lightly de-tokenized. Plain encyclopaedic prose for the language stage.",
     "license": "CC BY-SA 3.0 and GFDL", "source_url": "https://huggingface.co/datasets/Salesforce/wikitext",
     "size_hint": "6 MB download", "records_hint": "15,488 paragraphs", "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
     "repo": "Salesforce/wikitext", "filename": "wikitext-2-raw-v1/train-00000-of-00001.parquet", "format": "parquet", "parser": "wikitext",
     "split": "train", "synthetic": False},
    {"id": "simplewiki", "name": "Simple English Wikipedia · plain articles", "domain": "Language foundation", "stage": "language", "default": True,
     "description": "Every Simple English Wikipedia article (November 2023 dump): encyclopaedic prose written with short sentences and common words. Stubs under 200 characters are skipped.",
     "license": "CC BY-SA 3.0 and GFDL", "source_url": "https://huggingface.co/datasets/wikimedia/wikipedia",
     "size_hint": "157 MB download", "records_hint": "~240K articles", "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
     "repo": "wikimedia/wikipedia", "filename": "20231101.simple/train-00000-of-00001.parquet", "format": "parquet", "parser": "wiki_article",
     "split": "train", "synthetic": False},
    {"id": "wikitext103_a", "name": "WikiText-103 · Wikipedia prose, part 1", "domain": "Language foundation", "stage": "language", "default": True,
     "description": "First half of the WikiText-103 training split (raw): verified-good and featured Wikipedia articles as paragraphs. WikiText-2 is a small subset of it.",
     "license": "CC BY-SA 3.0 and GFDL", "source_url": "https://huggingface.co/datasets/Salesforce/wikitext",
     "size_hint": "157 MB download", "records_hint": "~400K paragraphs", "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
     "repo": "Salesforce/wikitext", "filename": "wikitext-103-raw-v1/train-00000-of-00002.parquet", "format": "parquet", "parser": "wikitext",
     "split": "train", "synthetic": False},
    {"id": "wikitext103_b", "name": "WikiText-103 · Wikipedia prose, part 2", "domain": "Language foundation", "stage": "language", "default": True,
     "description": "Second half of the WikiText-103 training split (raw), paragraphs tagged with article and section.",
     "license": "CC BY-SA 3.0 and GFDL", "source_url": "https://huggingface.co/datasets/Salesforce/wikitext",
     "size_hint": "157 MB download", "records_hint": "~400K paragraphs", "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
     "repo": "Salesforce/wikitext", "filename": "wikitext-103-raw-v1/train-00001-of-00002.parquet", "format": "parquet", "parser": "wikitext",
     "split": "train", "synthetic": False},
    # --- stage "instructions" -------------------------------------------------------------------
    {"id": "dolly15k", "name": "Dolly 15K · human instructions", "domain": "General instructions", "stage": "instructions", "default": True,
     "description": "Human-written instructions, extraction, summaries, brainstorming and question answering. Context is kept with each question.",
     "license": "CC BY-SA 3.0", "source_url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
     "size_hint": "13 MB download", "records_hint": "15,011 instructions", "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
     "repo": "databricks/databricks-dolly-15k", "filename": "databricks-dolly-15k.jsonl", "format": "jsonl", "parser": "dolly", "split": "train", "synthetic": False},
    {"id": "alpaca_cleaned", "name": "Alpaca Cleaned · short instructions", "domain": "General instructions", "stage": "instructions", "default": True,
     "description": "The Stanford Alpaca instruction set after community clean-up of hallucinated answers, empty outputs and bad formatting. Short, direct instruction/response pairs generated by text-davinci-003.",
     "license": "CC BY 4.0", "source_url": "https://huggingface.co/datasets/yahma/alpaca-cleaned",
     "size_hint": "44 MB download", "records_hint": "51,756 instructions", "revision": "12567cabf869d7c92e573c7c783905fc160e9639",
     "repo": "yahma/alpaca-cleaned", "filename": "alpaca_data_cleaned.json", "format": "json", "parser": "alpaca", "split": "train", "synthetic": True},
    {"id": "no_robots", "name": "No Robots · human-written SFT (non-commercial)", "domain": "General instructions", "stage": "instructions", "default": False,
     "description": "9.5K instructions and answers written by professional human annotators across ten categories. NON-COMMERCIAL licence (CC BY-NC 4.0), so it is not downloaded by default.",
     "license": "CC BY-NC 4.0 (non-commercial)", "source_url": "https://huggingface.co/datasets/HuggingFaceH4/no_robots",
     "size_hint": "10 MB download", "records_hint": "9,500 conversations", "revision": "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b",
     "repo": "HuggingFaceH4/no_robots", "filename": "data/train-00000-of-00001.parquet", "format": "parquet", "parser": "no_robots", "split": "train", "synthetic": False},
    {"id": "wizard143k", "name": "Evol-Instruct · expanded instructions", "domain": "Mixed instruction tasks", "stage": "instructions", "default": False,
     "description": "Optional full 143K evolved instruction/chat corpus from the WizardLM authors. Published synthetic data, with the source conversation structure retained.",
     "license": "MIT (publisher dataset card)", "source_url": "https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k",
     "size_hint": "375 MB download", "records_hint": "143K instruction conversations", "revision": "8a7d15a83028b5c93915677704f492839e2675f6",
     "repo": "WizardLMTeam/WizardLM_evol_instruct_V2_196k", "filename": "WizardLM_evol_instruct_V2_143k.json", "format": "json", "parser": "chat", "split": "train", "synthetic": True},
    # --- stage "knowledge" ----------------------------------------------------------------------
    {"id": "squad2", "name": "SQuAD 2 · grounded knowledge", "domain": "Knowledge and reading", "stage": "knowledge", "default": True,
     "description": "Full training split of human reading-comprehension questions. Includes source passages and explicit abstention for questions the passage cannot answer. Dev/test examples are excluded.",
     "license": "CC BY-SA 4.0", "source_url": "https://rajpurkar.github.io/SQuAD-explorer/",
     "size_hint": "42 MB download", "records_hint": "130,319 training questions", "revision": "eee5fdbf62f8613a7812b03419e6b29617b74fd1",
     "url": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/eee5fdbf62f8613a7812b03419e6b29617b74fd1/dataset/train-v2.0.json",
     "card_url": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/eee5fdbf62f8613a7812b03419e6b29617b74fd1/README.md",
     "filename": "train-v2.0.json", "format": "squad", "parser": "squad", "split": "train", "synthetic": False},
    {"id": "arc_easy", "name": "ARC Easy · grade-school science", "domain": "Science multiple choice", "stage": "knowledge", "default": True,
     "description": "Training split of the AI2 Reasoning Challenge easy set: real grade-school science exam questions. Converted to a question with lettered options and an answer such as \"B) food\".",
     "license": "CC BY-SA 4.0", "source_url": "https://huggingface.co/datasets/allenai/ai2_arc",
     "size_hint": "0.3 MB download", "records_hint": "2,251 questions", "revision": "210d026faf9955653af8916fad021475a3f00453",
     "repo": "allenai/ai2_arc", "filename": "ARC-Easy/train-00000-of-00001.parquet", "format": "parquet", "parser": "mcq", "split": "train", "synthetic": False},
    {"id": "arc_challenge", "name": "ARC Challenge · harder science", "domain": "Science multiple choice", "stage": "knowledge", "default": True,
     "description": "Training split of the AI2 Reasoning Challenge hard set: science exam questions that simple retrieval and word-association baselines failed. Lettered options with answer text.",
     "license": "CC BY-SA 4.0", "source_url": "https://huggingface.co/datasets/allenai/ai2_arc",
     "size_hint": "0.2 MB download", "records_hint": "1,118 questions", "revision": "210d026faf9955653af8916fad021475a3f00453",
     "repo": "allenai/ai2_arc", "filename": "ARC-Challenge/train-00000-of-00001.parquet", "format": "parquet", "parser": "mcq", "split": "train", "synthetic": False},
    {"id": "sciq", "name": "SciQ · science with evidence (non-commercial)", "domain": "Science multiple choice", "stage": "knowledge", "default": False,
     "description": "Crowdsourced science exam questions, each with a supporting passage and four options (shuffled deterministically). NON-COMMERCIAL licence (CC BY-NC 3.0), so it is not downloaded by default.",
     "license": "CC BY-NC 3.0 (non-commercial)", "source_url": "https://huggingface.co/datasets/allenai/sciq",
     "size_hint": "4 MB download", "records_hint": "11,679 questions", "revision": "2c94ad3e1aafab77146f384e23536f97a4849815",
     "repo": "allenai/sciq", "filename": "data/train-00000-of-00001.parquet", "format": "parquet", "parser": "sciq", "split": "train", "synthetic": False},
    # --- stage "reasoning" ----------------------------------------------------------------------
    {"id": "gsm8k", "name": "GSM8K · mathematical reasoning", "domain": "Math and reasoning", "stage": "reasoning", "default": True,
     "description": "Human-written grade-school word problems with worked solutions. Only the 7,473 training examples are imported; official test questions stay separate.",
     "license": "MIT", "source_url": "https://github.com/openai/grade-school-math",
     "size_hint": "4 MB download", "records_hint": "7,473 worked problems", "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
     "url": "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl",
     "card_url": "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/README.md",
     "license_url": "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE",
     "filename": "train.jsonl", "format": "jsonl", "parser": "gsm", "split": "train", "synthetic": False},
    {"id": "metamathqa", "name": "MetaMathQA · bounded math augmentation", "domain": "Math and reasoning", "stage": "reasoning", "default": True,
     "description": "Bootstrapped GSM8K/MATH questions with step-by-step answers written by GPT-3.5. Bounded to the first ~30 MB of the 395K file (whole records only).",
     "license": "MIT", "source_url": "https://huggingface.co/datasets/meta-math/MetaMathQA",
     "size_hint": "30 MB download (bounded prefix)", "records_hint": "29,846 worked problems", "revision": "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18",
     "repo": "meta-math/MetaMathQA", "filename": "MetaMathQA-395K.json", "format": "json", "parser": "metamath",
     "range_bytes": 30_000_000, "split": "train", "synthetic": True},
    {"id": "commonsenseqa", "name": "CommonsenseQA · everyday reasoning", "domain": "Commonsense multiple choice", "stage": "reasoning", "default": True,
     "description": "Training split of crowdsourced commonsense questions with five lettered options (built from ConceptNet relations). The answer is given as the letter and option text.",
     "license": "MIT", "source_url": "https://huggingface.co/datasets/tau/commonsense_qa",
     "size_hint": "1.2 MB download", "records_hint": "9,741 questions", "revision": "94630fe30dad47192a8546eb75f094926d47e155",
     "repo": "tau/commonsense_qa", "filename": "data/train-00000-of-00001.parquet", "format": "parquet", "parser": "mcq", "split": "train", "synthetic": False},
    # --- stage "code" ---------------------------------------------------------------------------
    {"id": "codealpaca20k", "name": "Code Alpaca · programming", "domain": "Coding", "stage": "code", "default": True,
     "description": "20K published programming instructions and answers generated by text-davinci-003. Treat code as unverified training text; nothing is executed.",
     "license": "CC BY 4.0 (publisher dataset card)", "source_url": "https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k",
     "size_hint": "8 MB download", "records_hint": "20,022 programming examples", "revision": "152bb5e9a29651266b018106053980070a0521a1",
     "repo": "sahil2801/CodeAlpaca-20k", "filename": "code_alpaca_20k.json", "format": "json", "parser": "code", "split": "train", "synthetic": True},
    {"id": "mbpp", "name": "MBPP · Python tasks", "domain": "Coding", "stage": "code", "default": True,
     "description": "Crowd-sourced entry-level Python problems (full config, training split only) with a reference solution. The task text and its assert tests form the prompt. Nothing is executed.",
     "license": "CC BY 4.0", "source_url": "https://huggingface.co/datasets/google-research-datasets/mbpp",
     "size_hint": "0.1 MB download", "records_hint": "374 problems", "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
     "repo": "google-research-datasets/mbpp", "filename": "full/train-00000-of-00001.parquet", "format": "parquet", "parser": "mbpp", "split": "train", "synthetic": False},
    # --- stage "conversation" -------------------------------------------------------------------
    {"id": "oasst1", "name": "OpenAssistant · conversation trees", "domain": "Conversation", "stage": "conversation", "default": True,
     "description": "Complete ready-for-export conversation trees from OpenAssistant. Preserves multilingual dialogue paths, source message IDs and human/synthetic provenance. Deleted or rejected branches are skipped.",
     "license": "Apache 2.0", "source_url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
     "size_hint": "~40 MB compressed download", "records_hint": "10,364 trees / 88,838 messages before conversion",
     "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b", "repo": "OpenAssistant/oasst1",
     "license_url": "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/fdf72ae0827c1cda404aff25b6603abec9e3399b/LICENSE?download=true",
     "filename": "2023-04-12_oasst_ready.trees.jsonl.gz", "format": "jsonl.gz", "parser": "oasst", "split": "ready_export", "synthetic": False},
    {"id": "ultrachat_a", "name": "UltraChat 200K · multi-turn chats, part 1", "domain": "Conversation", "stage": "conversation", "default": True,
     "description": "First of three shards of the filtered UltraChat SFT split: multi-turn English conversations about the world, writing and assistance, written by GPT-3.5.",
     "license": "MIT", "source_url": "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k",
     "size_hint": "244 MB download", "records_hint": "~69K conversations", "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
     "repo": "HuggingFaceH4/ultrachat_200k", "filename": "data/train_sft-00000-of-00003-a3ecf92756993583.parquet", "format": "parquet",
     "parser": "ultrachat", "split": "train_sft", "synthetic": True},
    # --- stage "problems": questions with an answer key a rule checks (Post-training, round 2) -------------
    {"id": "gsm8k_check", "name": "GSM8K · word problems with a numeric key", "domain": "Checkable problems", "stage": "problems", "default": True,
     "description": "The 7,473 GSM8K training problems, each stored with its final number as the key. Checked by the final number in the answer.",
     "license": "MIT", "source_url": "https://github.com/openai/grade-school-math",
     "size_hint": "4 MB download", "records_hint": "7,473 problems", "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
     "url": "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl",
     "card_url": "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/README.md",
     "filename": "train.jsonl", "format": "jsonl", "parser": "gsm_check", "split": "train", "synthetic": False},
    {"id": "svamp_check", "name": "SVAMP · one-step arithmetic stories", "domain": "Checkable problems", "stage": "problems", "default": True,
     "description": "Short arithmetic word problems built to defeat shallow pattern matching (training split). Checked by the final number.",
     "license": "MIT", "source_url": "https://huggingface.co/datasets/ChilleD/SVAMP",
     "size_hint": "0.1 MB download", "records_hint": "700 problems", "revision": "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef",
     "repo": "ChilleD/SVAMP", "filename": "data/train-00000-of-00001.parquet", "format": "parquet", "parser": "svamp_check", "split": "train", "synthetic": False},
    {"id": "metamath_check", "name": "MetaMathQA · numeric-answer problems", "domain": "Checkable problems", "stage": "problems", "default": True,
     "description": "The bounded MetaMathQA prefix, keeping only problems whose stated answer is a plain number. Checked by the final number.",
     "license": "MIT", "source_url": "https://huggingface.co/datasets/meta-math/MetaMathQA",
     "size_hint": "30 MB download (bounded prefix)", "records_hint": "~25K problems", "revision": "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18",
     "repo": "meta-math/MetaMathQA", "filename": "MetaMathQA-395K.json", "format": "json", "parser": "metamath_check",
     "range_bytes": 30_000_000, "split": "train", "synthetic": True},
    {"id": "aqua_check", "name": "AQuA-RAT · algebra multiple choice", "domain": "Checkable problems", "stage": "problems", "default": True,
     "description": "Algebraic word problems with five lettered options (raw training split). Checked by the option letter.",
     "license": "Apache 2.0", "source_url": "https://huggingface.co/datasets/deepmind/aqua_rat",
     "size_hint": "25 MB download", "records_hint": "97,467 problems", "revision": "33301c6a050c96af81f63cad5562cb5363e88971",
     "repo": "deepmind/aqua_rat", "filename": "raw/train-00000-of-00001.parquet", "format": "parquet", "parser": "aqua_check", "split": "train", "synthetic": False},
    {"id": "arc_check", "name": "ARC Easy · science questions with a key", "domain": "Checkable problems", "stage": "problems", "default": True,
     "description": "ARC Easy training questions as problems. Checked by the option letter.",
     "license": "CC BY-SA 4.0", "source_url": "https://huggingface.co/datasets/allenai/ai2_arc",
     "size_hint": "0.3 MB download", "records_hint": "2,251 problems", "revision": "210d026faf9955653af8916fad021475a3f00453",
     "repo": "allenai/ai2_arc", "filename": "ARC-Easy/train-00000-of-00001.parquet", "format": "parquet", "parser": "mcq_check", "split": "train", "synthetic": False},
    {"id": "arc_challenge_check", "name": "ARC Challenge · harder science with a key", "domain": "Checkable problems", "stage": "problems", "default": True,
     "description": "ARC Challenge training questions as problems. Checked by the option letter.",
     "license": "CC BY-SA 4.0", "source_url": "https://huggingface.co/datasets/allenai/ai2_arc",
     "size_hint": "0.2 MB download", "records_hint": "1,118 problems", "revision": "210d026faf9955653af8916fad021475a3f00453",
     "repo": "allenai/ai2_arc", "filename": "ARC-Challenge/train-00000-of-00001.parquet", "format": "parquet", "parser": "mcq_check", "split": "train", "synthetic": False},
    {"id": "commonsenseqa_check", "name": "CommonsenseQA · everyday questions with a key", "domain": "Checkable problems", "stage": "problems", "default": True,
     "description": "CommonsenseQA training questions as problems. Checked by the option letter.",
     "license": "MIT", "source_url": "https://huggingface.co/datasets/tau/commonsense_qa",
     "size_hint": "1.2 MB download", "records_hint": "9,741 problems", "revision": "94630fe30dad47192a8546eb75f094926d47e155",
     "repo": "tau/commonsense_qa", "filename": "data/train-00000-of-00001.parquet", "format": "parquet", "parser": "mcq_check", "split": "train", "synthetic": False},
    {"id": "qasc_check", "name": "QASC · two-fact science questions", "domain": "Checkable problems", "stage": "problems", "default": True,
     "description": "Grade-school science questions with eight lettered options, each answerable by combining two facts (training split). Checked by the option letter.",
     "license": "CC BY 4.0", "source_url": "https://huggingface.co/datasets/allenai/qasc",
     "size_hint": "2 MB download", "records_hint": "8,134 problems", "revision": "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070",
     "repo": "allenai/qasc", "filename": "data/train-00000-of-00001.parquet", "format": "parquet", "parser": "mcq_check", "split": "train", "synthetic": False},
    {"id": "hellaswag_check", "name": "HellaSwag · pick the sensible continuation", "domain": "Checkable problems", "stage": "problems", "default": True,
     "description": "Everyday activity descriptions with four possible continuations, three of them adversarially written (training split). Checked by the option letter.",
     "license": "MIT", "source_url": "https://huggingface.co/datasets/Rowan/hellaswag",
     "size_hint": "24 MB download", "records_hint": "39,905 problems", "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
     "repo": "Rowan/hellaswag", "filename": "data/train-00000-of-00001.parquet", "format": "parquet", "parser": "hellaswag_check", "split": "train", "synthetic": False},
    {"id": "boolq_check", "name": "BoolQ · yes/no questions about a passage", "domain": "Checkable problems", "stage": "problems", "default": True,
     "description": "Naturally occurring yes/no questions with the Wikipedia passage that answers them (training split), as A) yes / B) no. Checked by the option letter.",
     "license": "CC BY-SA 3.0", "source_url": "https://huggingface.co/datasets/google/boolq",
     "size_hint": "4 MB download", "records_hint": "9,427 problems", "revision": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
     "repo": "google/boolq", "filename": "data/train-00000-of-00001.parquet", "format": "parquet", "parser": "boolq_check", "split": "train", "synthetic": False},
    {"id": "winogrande_check", "name": "WinoGrande · who does the blank refer to", "domain": "Checkable problems", "stage": "problems", "default": False,
     "description": "Fill-in-the-blank pronoun puzzles with two options (XL training split). Not downloaded by default: the Hugging Face card states no licence.",
     "license": "Not stated on the dataset card", "source_url": "https://huggingface.co/datasets/allenai/winogrande",
     "size_hint": "2 MB download", "records_hint": "40,398 problems", "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
     "repo": "allenai/winogrande", "filename": "winogrande_xl/train-00000-of-00001.parquet", "format": "parquet", "parser": "winogrande_check", "split": "train", "synthetic": False},
]
STAGES = ("language", "instructions", "knowledge", "reasoning", "code", "conversation", "problems")
STARTER_IDS = ("dolly15k", "squad2", "codealpaca20k", "gsm8k", "oasst1")


def default_catalog_ids():
    """Ids downloaded by default (commercially usable, bounded), in catalogue/stage order."""
    return [entry["id"] for entry in CATALOG if entry.get("default")]
FORMAT_VERSION = 2  # 2: line endings normalised to \n; older converted files are rebuilt from the cached source
MAX_DOWNLOAD_BYTES = 1_000_000_000
MAX_RECORD_CHARS = 16_000_000
_DOWNLOAD_LOCK = threading.Lock()


class DownloadCancelled(Exception):
    pass


def _cancel(stop):
    if stop is not None and stop.is_set():
        raise DownloadCancelled("Download cancelled. Completed source files are cached for the next attempt.")


def file_sha256(path, stop=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            _cancel(stop)
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url, path, stop=None, emit=None, maximum=MAX_DOWNLOAD_BYTES, range_bytes=None):
    """A single atomic, bounded HTTPS transfer; interrupted .part files are removed.

    With range_bytes only the first N bytes are requested (HTTP Range) and kept; if a server
    ignores the header the stream is cut at N bytes. The receipt hash covers exactly those bytes.
    """
    _cancel(stop)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Groundwork-AI-Studio/2.0", "Accept-Encoding": "identity"}
    if range_bytes:
        maximum = min(maximum, int(range_bytes))
        headers["Range"] = f"bytes=0-{int(range_bytes) - 1}"
    request = urllib.request.Request(url, headers=headers)
    completed, last_emit = 0, 0.0
    try:
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            if range_bytes:
                total = min(total, int(range_bytes)) if total else 0
            elif total > maximum:
                raise ValueError("Source exceeds the catalogue download size limit.")
            digest = hashlib.sha256()
            while True:
                _cancel(stop)
                chunk = response.read(1024 * 512 if not range_bytes else min(1024 * 512, int(range_bytes) - completed))
                if not chunk:
                    break
                completed += len(chunk)
                if completed > maximum:
                    raise ValueError("Source exceeds the catalogue download size limit.")
                output.write(chunk)
                digest.update(chunk)
                if emit and time.monotonic() - last_emit > .2:
                    emit({"text": f"Downloading {path.name} · {completed / 1e6:.1f} MB", "completed": completed, "total": total})
                    last_emit = time.monotonic()
                if range_bytes and completed >= int(range_bytes):
                    break
            if total and completed != total:
                raise OSError(f"Incomplete transfer: expected {total} bytes, received {completed}.")
        _cancel(stop)
        replace_file(temporary, path)
        return {"url": url, "filename": path.name, "bytes": completed, "sha256": digest.hexdigest()}
    finally:
        temporary.unlink(missing_ok=True)


_JSON_SPACE = re.compile(r"[ \t\n\r]*")


def iter_json_array(stream, chunk_size=65536, truncated=False):
    """Incrementally parse a top-level JSON array without a third-party parser.

    truncated=True accepts a byte-bounded prefix of an array: only whole records are yielded and
    the cut-off final record is dropped.
    """
    decoder = json.JSONDecoder()
    buffer, pos, eof = "", 0, False

    def read_more(size=chunk_size):
        nonlocal buffer, pos, eof
        chunk = stream.read(size)
        if not chunk:
            eof = True
        buffer, pos = buffer[pos:] + chunk, 0  # drop consumed text; records are decoded from `pos`

    read_more()
    buffer = buffer.lstrip("\ufeff \r\n\t")
    if not buffer.startswith("["):
        raise ValueError("Expected a JSON array.")
    pos = 1
    needs_item = False
    while True:
        pos = _JSON_SPACE.match(buffer, pos).end()
        if pos == len(buffer) and not eof:
            read_more()
            continue
        if buffer.startswith("]", pos):
            if needs_item:
                raise ValueError("Trailing comma in source JSON array.")
            rest = buffer[pos + 1:] + stream.read()
            if rest.strip():
                raise ValueError("Unexpected content after JSON array.")
            return
        try:
            value, end = decoder.raw_decode(buffer, pos)
        except json.JSONDecodeError as exc:
            if eof:
                if truncated:
                    return
                raise ValueError("Incomplete or invalid source JSON array.") from exc
            if len(buffer) - pos > MAX_RECORD_CHARS:
                raise ValueError("A source JSON record exceeds the 16 MB parser limit.") from exc
            # Read at least as much again as is pending, so a large record is re-decoded O(log n) times.
            read_more(max(chunk_size, len(buffer) - pos))
            continue
        yield value
        pos = _JSON_SPACE.match(buffer, end).end()
        while pos == len(buffer) and not eof:
            read_more()
            pos = _JSON_SPACE.match(buffer, pos).end()
        if buffer.startswith(",", pos):
            pos += 1
            needs_item = True
        elif buffer.startswith("]", pos):
            needs_item = False
        elif truncated and pos == len(buffer):
            return
        else:
            raise ValueError("Expected a comma or closing bracket in source JSON array.")


STORY_DELIMITER = "<|endoftext|>"


def iter_stories(stream, delimiter=STORY_DELIMITER, chunk_size=1 << 20):
    """Split a plain-text corpus on its end-of-text marker without loading it whole.

    Text after the last marker is dropped: it is either a byte-bounded prefix's cut-off story or,
    for a whole file fetched without Content-Length, possibly a truncated one.
    """
    buffer = ""
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        buffer += chunk
        *finished, buffer = buffer.split(delimiter)
        for story in finished:
            if story.strip():
                yield story.strip()
        if len(buffer) > MAX_RECORD_CHARS:
            raise ValueError("A source text record exceeds the 16 MB parser limit.")


def iter_parquet(path, batch_size=2048):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
        raise RuntimeError("Parquet datasets need pyarrow: pip install pyarrow") from exc
    parquet = pq.ParquetFile(str(path))
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()
    finally:
        parquet.close()


_WIKI_FIXES = ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" ) ", ") "), (" ( ", " ("),
               (" , ", ", "), (" . ", ". "), (" ; ", "; "), (" : ", ": "), (" 's", "'s"), (" n't", "n't"), (" 're", "'re"), (" 've", "'ve"), (" 'd", "'d"),
               (" 'll", "'ll"), (" 'm", "'m"), (" ? ", "? "), (" ! ", "! "))


def clean_wikitext(text):
    """Undo WikiText's token-spacing so paragraphs read as ordinary prose."""
    text = " " + text.strip() + " "
    for old, new in _WIKI_FIXES:
        text = text.replace(old, new)
    text = re.sub(r"\s+([.,;:!?)])(?=\s|$)", r"\1", text)
    text = re.sub(r'" ([^"]+?) "', r'"\1"', text)
    return re.sub(r"\s+", " ", text).strip()


def wikitext_paragraphs(rows, minimum_chars=80):
    """Turn WikiText line rows into paragraph records tagged with article and section."""
    title, section = "", ""
    for row in rows:
        line = str(row.get("text", "")).strip()
        if not line:
            continue
        if line.startswith("=") and line.endswith("="):
            name = line.strip("= ").strip()
            if line.startswith("= ="):
                section = name
            else:
                title, section = name, ""
            continue
        paragraph = clean_wikitext(line)
        if len(paragraph) >= minimum_chars:
            yield {"text": paragraph, "title": title, "section": section}


def source_records(path, entry):
    path = Path(path)
    if entry["format"] == "parquet":
        rows = iter_parquet(path)
        yield from (wikitext_paragraphs(rows) if entry.get("parser") == "wikitext" else rows)
        return
    if entry["format"] == "txt_stories":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for story in iter_stories(stream):
                yield {"text": story}
        return
    if entry["format"] == "squad":
        # SQuAD's fixed 42 MB nested object is bounded, unlike expandable chat arrays.
        if path.stat().st_size > 100_000_000:
            raise ValueError("SQuAD source is unexpectedly large.")
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        for article in payload["data"]:
            for paragraph in article["paragraphs"]:
                for question in paragraph["qas"]:
                    yield {**question, "context": paragraph["context"], "title": article.get("title", "")}
        return
    opener = gzip.open if entry["format"].endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        if entry["format"] == "json":
            yield from iter_json_array(stream, truncated=bool(entry.get("range_bytes")))
        else:
            line_number = 0
            while True:
                line = stream.readline(MAX_RECORD_CHARS + 1)
                if not line:
                    break
                line_number += 1
                if len(line) > MAX_RECORD_CHARS:
                    raise ValueError(f"Source line {line_number} exceeds the parser size limit.")
                if line.strip():
                    try:
                        yield json.loads(line)
                    except ValueError as exc:
                        raise ValueError(f"Invalid source JSON on line {line_number}.") from exc


def convert_record(raw, entry):
    """Yield canonical records while preserving task context and source ancestry."""
    parser = entry["parser"]
    if parser == "oasst":
        tree_id = raw.get("message_tree_id", "")
        stack = [(raw.get("prompt", {}), [], False)]
        while stack:
            node, ancestors, synthetic = stack.pop()
            if node.get("deleted") or node.get("review_result") is False:
                continue
            role = {"prompter": "user", "assistant": "assistant"}.get(node.get("role"))
            if role is None or not node.get("text", "").strip():
                continue
            turns = ancestors + [{"role": role, "content": node["text"]}]
            synthetic = synthetic or truthy(node.get("synthetic"))
            if role == "assistant":
                yield {"messages": turns, "intent": "conversation", "synthetic": synthetic,
                       "sourceId": node.get("message_id", ""), "groupId": f"oasst1:{tree_id}",
                       "metadata": {"language": node.get("lang", ""), "tree_id": tree_id, "rank": node.get("rank")}}
            for child in reversed(node.get("replies", [])):
                stack.append((child, turns, synthetic))
        return
    if parser == "squad":
        impossible = bool(raw.get("is_impossible"))
        answers = raw.get("answers", [])
        if not impossible and not answers:
            return
        context = raw["context"]
        yield {"user": "Answer using only the passage. If the passage does not contain the answer, say so.\n\nPassage:\n" + context + "\n\nQuestion: " + raw["question"],
               "assistant": "The passage does not contain enough information to answer this question." if impossible else answers[0]["text"],
               "intent": "unanswerable" if impossible else "reading_comprehension", "sourceId": raw.get("id", ""),
               "groupId": "squad2:" + hashlib.sha256(context.encode()).hexdigest(),
               "metadata": {"title": raw.get("title", ""), "is_impossible": impossible}}
        return
    if parser == "dolly":
        prompt = raw["instruction"]
        if raw.get("context", "").strip():
            prompt += "\n\nContext:\n" + raw["context"]
        yield {"user": prompt, "assistant": raw["response"], "intent": raw.get("category", "instruction")}
    elif parser == "code":
        value = dict(raw)
        if value.get("input", "").strip().lower() in {"<noinput>", "<no input>"}:
            value["input"] = ""
        yield {**value, "intent": "coding"}
    elif parser == "gsm":
        yield {"user": raw["question"], "assistant": raw["answer"], "intent": "math_reasoning"}
    elif parser == "story":
        yield {"text": raw["text"], "intent": "story"}
    elif parser == "wikitext":
        yield {"text": raw["text"], "intent": "encyclopedia", "groupId": "wikitext2:" + raw.get("title", ""),
               "metadata": {"title": raw.get("title", ""), "section": raw.get("section", "")}}
    elif parser == "alpaca":
        instruction, extra, output = (str(raw.get(key) or "").strip() for key in ("instruction", "input", "output"))
        if instruction and output:
            yield {"user": instruction + ("\n\nInput:\n" + extra if extra else ""), "assistant": output, "intent": "instruction"}
    elif parser == "no_robots":
        yield {"messages": raw["messages"], "intent": str(raw.get("category") or "instruction").lower(), "sourceId": raw.get("prompt_id", ""),
               "groupId": "no_robots:" + str(raw.get("prompt_id", ""))}
    elif parser == "metamath":
        query, response = str(raw.get("query") or "").strip(), str(raw.get("response") or "").strip()
        if query and response:
            yield {"user": query, "assistant": response, "intent": "math_reasoning", "metadata": {"type": raw.get("type", "")}}
    elif parser == "mbpp":
        tests = [str(test).strip() for test in raw.get("test_list") or [] if str(test).strip()]
        prompt = str(raw["text"]).strip() + ("\n\nYour code must pass these tests:\n" + "\n".join(tests) if tests else "")
        code = str(raw["code"]).replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ").strip()
        yield {"user": prompt, "assistant": code, "intent": "coding", "sourceId": f"mbpp:{raw.get('task_id', '')}"}
    elif parser in {"mcq", "sciq"}:
        converted = _multiple_choice(raw, parser)
        if converted:
            yield converted
    elif parser == "wiki_article":
        text = str(raw.get("text") or "").strip()
        if len(text) >= 200:
            yield {"text": text, "intent": "encyclopedia", "sourceId": str(raw.get("id", "")),
                   "groupId": "simplewiki:" + str(raw.get("id", "")), "metadata": {"title": raw.get("title", "")}}
    elif parser == "ultrachat":
        yield {"messages": raw.get("messages") or [], "intent": "conversation", "sourceId": str(raw.get("prompt_id", "")),
               "groupId": "ultrachat:" + str(raw.get("prompt_id", ""))}
    elif parser.endswith("_check"):
        converted = _checkable(raw, parser)
        if converted:
            yield converted
    else:
        yield raw


def _unix_newlines(value):
    """CRLF and lone CR become LF in every string, so byte-level tokenizers never see \\r."""
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    if isinstance(value, dict):
        return {key: _unix_newlines(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unix_newlines(item) for item in value]
    return value


_LETTERS = "ABCDEFGHIJ"


def _multiple_choice(raw, parser):
    """Question + lettered options -> 'B) option text'; deterministic and loss-free."""
    question = str(raw.get("question") or "").strip()
    if parser == "sciq":
        options = [str(raw.get(key) or "").strip() for key in ("distractor1", "distractor2", "distractor3", "correct_answer")]
        correct = options[3]
        # Stable shuffle seeded by the question so the answer letter is not always D.
        random.Random(int(hashlib.sha256(question.encode()).hexdigest()[:12], 16)).shuffle(options)
        index = options.index(correct) if correct else -1
        background = str(raw.get("support") or "").strip()
        source_id = "sciq:" + hashlib.sha256(question.encode()).hexdigest()[:16]
        intro = "Answer the science question. Reply with the letter and the option text.\n\n" + ("Background:\n" + background + "\n\n" if background else "")
        intent = "science_qa"
    else:
        choices = raw.get("choices") or {}
        labels, options = list(choices.get("label") or []), [str(text).strip() for text in choices.get("text") or []]
        key = str(raw.get("answerKey") or "").strip()
        index = labels.index(key) if key in labels else -1
        source_id = str(raw.get("id", ""))
        intro = "Answer the question. Reply with the letter and the option text.\n\n"
        intent = "commonsense_qa" if raw.get("question_concept") is not None else "science_qa"
    if not question or len(options) < 2 or len(options) > len(_LETTERS) or not 0 <= index < len(options) or not all(options):
        return None
    lines = [f"{_LETTERS[i]}) {text}" for i, text in enumerate(options)]
    return {"user": f"{intro}Question: {question}\n\n" + "\n".join(lines), "assistant": lines[index], "intent": intent,
            "sourceId": source_id}


NUMBER_ASK = "\n\nGive the final answer as a number."
CHOICE_ASK = "Reply with the letter and the option text.\n\n"
PLAIN_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _number_key(text):
    """A plain number as a key ('1,200' -> '1200', '145.0' -> '145'), else None: fractions, LaTeX and words are not keys."""
    text = str(text if text is not None else "").strip().rstrip(".")
    if not PLAIN_NUMBER.fullmatch(text):
        return None
    value = float(text.replace(",", ""))
    return str(int(value)) if value == int(value) else repr(value)


def _keyed(prompt, answer, rule, key, intent, source_id=""):
    """A problem row: the key text is the stored answer; rule and key travel in metadata, which normalize keeps."""
    return {"user": prompt, "assistant": answer, "intent": intent, "sourceId": source_id,
            "metadata": {"rule": rule, "params": {"answer": key}}}


def _lettered(intro, question, options, index, intent, source_id=""):
    if not question or not 2 <= len(options) <= len(_LETTERS) or not 0 <= index < len(options) or not all(options):
        return None
    lines = [f"{_LETTERS[i]}) {text}" for i, text in enumerate(options)]
    return _keyed(f"{intro}{CHOICE_ASK}{question}\n\n" + "\n".join(lines), lines[index], "choice", _LETTERS[index], intent, source_id)


def _checkable(raw, parser):
    """Source record -> a problem with an answer key (studio_rules.KEYED), or None when it has no clean key."""
    if parser == "gsm_check":
        key = _number_key(str(raw.get("answer", "")).rpartition("####")[2])
        question = str(raw.get("question") or "").strip()
        return _keyed(question + NUMBER_ASK, key, "final_number", key, "math_problem") if key and question else None
    if parser == "svamp_check":
        key = _number_key(raw.get("Answer"))
        question = str(raw.get("question_concat") or f"{raw.get('Body', '')} {raw.get('Question', '')}").strip()
        return _keyed(question + NUMBER_ASK, key, "final_number", key, "math_problem", str(raw.get("ID", ""))) if key and question else None
    if parser == "metamath_check":
        query, response = str(raw.get("query") or "").strip(), str(raw.get("response") or "")
        key = _number_key(response.rpartition("The answer is:")[2]) if "The answer is:" in response else None
        return _keyed(query + NUMBER_ASK, key, "final_number", key, "math_problem") if key and query else None
    if parser == "aqua_check":
        options = [re.sub(r"^\s*[A-E]\s*\)\s*", "", str(o)).strip() for o in raw.get("options") or []]
        key = str(raw.get("correct") or "").strip().upper()
        return _lettered("Solve the problem and pick the option. ", str(raw.get("question") or "").strip(), options,
                         _LETTERS.find(key) if key else -1, "math_problem")
    if parser == "mcq_check":
        choices = raw.get("choices") or {}
        labels, options = list(choices.get("label") or []), [str(t).strip() for t in choices.get("text") or []]
        key = str(raw.get("answerKey") or "").strip()
        return _lettered("Answer the question. ", str(raw.get("question") or "").strip(), options,
                         labels.index(key) if key in labels else -1, "multiple_choice", str(raw.get("id", "")))
    if parser == "hellaswag_check":
        label = str(raw.get("label") or "").strip()
        context = ": ".join(p for p in (str(raw.get("activity_label") or "").strip(), str(raw.get("ctx") or "").strip()) if p)
        return _lettered("Pick the most sensible continuation. ", context, [str(e).strip() for e in raw.get("endings") or []],
                         int(label) if label.isdigit() else -1, "multiple_choice", str(raw.get("source_id", "")))
    if parser == "boolq_check":
        question, passage, answer = str(raw.get("question") or "").strip(), str(raw.get("passage") or "").strip(), raw.get("answer")
        if not question or not passage or not isinstance(answer, bool):
            return None
        return _lettered("Answer using the passage. ", f"Passage:\n{passage}\n\nQuestion: {question[0].upper()}{question[1:]}?",
                         ["yes", "no"], 0 if answer else 1, "reading_problem")
    if parser == "winogrande_check":
        answer = str(raw.get("answer") or "").strip()
        options = [str(raw.get("option1") or "").strip(), str(raw.get("option2") or "").strip()]
        return _lettered("Which option fills the blank (_)? ", str(raw.get("sentence") or "").strip(), options,
                         int(answer) - 1 if answer in ("1", "2") else -1, "multiple_choice")
    return None


def _entry(catalog_id):
    entry = next((item for item in CATALOG if item["id"] == catalog_id), None)
    if entry is None:
        raise ValueError("Unknown catalogue dataset.")
    return entry


def _read_state(path, default):
    """Read a JSON state file. Missing or unreadable state returns `default`, so the caller rebuilds it."""
    path = Path(path)
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        _log.warning("Ignoring unreadable %s (%s); it will be rebuilt.", path, error)
        return default
    if not isinstance(value, type(default)):
        _log.warning("Ignoring malformed %s; it will be rebuilt.", path)
        return default
    return value


def download_dataset(catalog_id, root=ROOT, stop=None, emit=None):
    """Fetch a complete pinned corpus and return its verified normalized manifest."""
    entry = _entry(catalog_id)
    root = Path(root)
    folder = root / "data" / "downloads" / catalog_id
    folder.mkdir(parents=True, exist_ok=True)
    normalized = folder / "records.jsonl"
    manifest_path = folder / "manifest.json"
    _cancel(stop)
    # One catalog transfer at a time protects .part/manifests from concurrent writes.
    if not _DOWNLOAD_LOCK.acquire(blocking=False):
        raise RuntimeError("Another catalogue download is already running.")
    try:
        if manifest_path.exists() and normalized.exists():
            cached = _read_state(manifest_path, {})
            if (cached.get("status") == "complete" and cached.get("format_version") == FORMAT_VERSION
                    and cached.get("revision") == entry["revision"] and cached.get("sha256") == file_sha256(normalized, stop)):
                if emit:
                    emit({"text": f"Verified cached {entry['name']} · {cached['records']:,} records", "completed": cached["records"], "total": cached["records"]})
                return {**cached, "path": str(normalized), "manifest_path": str(manifest_path), "cached": True}
            atomic_json(manifest_path, {**cached, "status": "needs_rebuild"})
        raw_path = folder / Path(entry["filename"]).name
        receipt_path = folder / "source.json"
        source_url = entry.get("url") or _hf(entry["repo"], entry["revision"], entry["filename"])
        receipt = _read_state(receipt_path, {})
        range_bytes = entry.get("range_bytes")
        if not (raw_path.exists() and receipt.get("url") == source_url and receipt.get("range_bytes") == range_bytes
                and receipt.get("sha256") == file_sha256(raw_path, stop)):
            receipt = _fetch(source_url, raw_path, stop, emit, **({"range_bytes": range_bytes} if range_bytes else {}))
            if range_bytes:
                receipt["range_bytes"] = range_bytes
            atomic_json(receipt_path, receipt)
        if entry.get("sha256") and receipt["sha256"] != entry["sha256"]:
            raw_path.unlink(missing_ok=True)
            receipt_path.unlink(missing_ok=True)
            raise ValueError(f"{entry['name']} does not match its pinned checksum; the download was discarded.")
        card_url = entry.get("card_url") or _hf(entry["repo"], entry["revision"], "README.md")
        if not (folder / "SOURCE_README.md").exists():
            try:
                _fetch(card_url, folder / "SOURCE_README.md", stop, emit, maximum=5_000_000)
            except (OSError, ValueError):  # the card is optional
                pass
        license_file = None
        if entry.get("license_url"):
            license_file = "SOURCE_LICENSE.txt"
            if not (folder / license_file).exists():
                try:  # like the card, retried on the next download; the manifest records the gap
                    _fetch(entry["license_url"], folder / license_file, stop, emit, maximum=5_000_000)
                except (OSError, ValueError):
                    license_file = "unavailable"
        records, invalid, duplicate = 0, 0, 0
        seen = set()
        temp = normalized.with_suffix(".jsonl.part")
        try:
            with temp.open("w", encoding="utf-8", newline="\n") as stream:
                limit = entry.get("max_records")
                for index, raw in enumerate(source_records(raw_path, entry)):
                    _cancel(stop)
                    if limit and records >= limit:
                        break
                    for converted in convert_record(raw, entry):
                        converted = _unix_newlines(converted)
                        row = normalize({**converted, "synthetic": converted.get("synthetic", entry["synthetic"]),
                                         "quality": "Review", "source": f"catalog:{catalog_id}", "license": entry["license"],
                                         "source_url": entry["source_url"], "split": entry["split"]}, source=f"catalog:{catalog_id}")
                        if row is None:
                            invalid += 1
                            continue
                        key = fingerprint(row)
                        if key in seen:
                            duplicate += 1
                            continue
                        seen.add(key)
                        row["sourceId"] = str(converted.get("sourceId", converted.get("id", index)))
                        row["groupId"] = converted.get("groupId") or f"{catalog_id}:{key}"
                        row["provenance"] = {"catalog_id": catalog_id, "revision": entry["revision"], "source_id": row["sourceId"], "split": entry["split"]}
                        for name in ("id", "datasetId", "updatedAt"):
                            row.pop(name, None)
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        records += 1
                    if emit and index % 2000 == 0:
                        emit({"text": f"Preparing {entry['name']} · {records:,} records", "completed": index, "total": 0})
            if records == 0:
                raise ValueError("This download contained no usable training records.")
            _cancel(stop)
            replace_file(temp, normalized)
        except (ValueError, EOFError):
            # A source that ends mid-record (e.g. truncated without Content-Length) must not stay cached.
            receipt_path.unlink(missing_ok=True)
            raise
        finally:
            temp.unlink(missing_ok=True)
        manifest = {"format_version": FORMAT_VERSION, "status": "complete", "catalog_id": catalog_id, "name": entry["name"],
                    "description": entry["description"], "path": str(normalized), "records": records,
                    "bytes": normalized.stat().st_size, "sha256": file_sha256(normalized, stop),
                    "source_url": entry["source_url"], "license": entry["license"], "revision": entry["revision"],
                    "split": entry["split"], "stage": entry.get("stage", ""), "downloadedAt": now(), "invalid": invalid, "duplicates": duplicate,
                    "source": receipt, "manifest_path": str(manifest_path), "synthetic": entry["synthetic"],
                    "license_file": license_file}
        atomic_json(manifest_path, manifest)
        if emit:
            emit({"text": f"Ready · {records:,} {entry['name']} records", "completed": records, "total": records})
        return manifest
    finally:
        _DOWNLOAD_LOCK.release()


def list_downloads(root=ROOT):
    """Fast inventory for GUI startup; full checksum validation happens on reuse."""
    results = []
    for entry in CATALOG:
        folder = Path(root) / "data" / "downloads" / entry["id"]
        try:
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            path = folder / "records.jsonl"
            if manifest.get("status") == "complete" and path.stat().st_size == manifest.get("bytes"):
                results.append({**manifest, "path": str(path), "manifest_path": str(folder / "manifest.json")})
        except (OSError, ValueError):
            pass
    return results


def register_download(workspace, result, stop=None, emit=None):
    """Index a verified download in SQLite; repeated registration is cheap.

    Rows are written in batched transactions under a staging dataset id without the workspace lock,
    then swapped in with one short transaction; the lock is only taken to update workspace metadata.
    """
    catalog_id = result["catalog_id"]
    _entry(catalog_id)
    dataset_id = f"catalog_{catalog_id}"
    existing = workspace._dataset(dataset_id)
    stale = existing is not None and existing.get("sourceSha256") != result.get("sha256")
    with workspace._connect() as connection:
        count = connection.execute("SELECT COUNT(*) FROM records WHERE dataset_id=?", (dataset_id,)).fetchone()[0]
    if existing is not None and not stale and count == result["records"]:
        return dataset_id
    _cancel(stop)
    path = Path(result["path"])
    if result.get("status") != "complete" or not path.is_file() or file_sha256(path, stop) != result.get("sha256"):
        raise ValueError("The downloaded corpus failed integrity verification. Download it again.")
    if workspace.path.exists():
        backup = workspace.path.with_name("workspace.before-catalog-" + time.strftime("%Y%m%d-%H%M%S") + ".json")
        if not backup.exists():
            shutil.copy2(workspace.path, backup)
        for old in sorted(workspace.path.parent.glob("workspace.before-catalog-*.json"))[:-3]:
            old.unlink(missing_ok=True)
    if count != result["records"] or stale:
        # One transaction swaps the old rows for the new ones: WAL readers keep seeing the old set until the commit,
        # and a cancel or failure rolls back to it. No staging copy to rename and one sync instead of one per batch.
        insert = "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)"
        with workspace._connect() as connection:
            connection.execute("PRAGMA synchronous=NORMAL")  # WAL stays consistent; a power cut may lose this import only
            try:
                connection.execute("DELETE FROM records WHERE dataset_id IN (?,?)", (dataset_id, dataset_id + ".importing"))  # + an older version's staging rows
                count, batch, stamp = 0, [], now()
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        _cancel(stop)
                        # records.jsonl was normalized when it was written and its checksum verified above,
                        # so a row only needs the fields download_dataset stripped: id, dataset and time.
                        row = json.loads(line)
                        if not isinstance(row, dict) or not (row.get("text") or row.get("messages")):
                            raise ValueError("A normalized catalogue row became invalid.")
                        row.update(id=uid("ex"), datasetId=dataset_id, updatedAt=stamp)
                        batch.append(workspace._db_row(row))
                        count += 1
                        if len(batch) >= 5000:
                            connection.executemany(insert, batch)
                            batch.clear()
                        if emit and count % 5000 == 0:
                            emit({"text": f"Indexing {result['name']} · {count:,}/{result['records']:,}", "completed": count, "total": result["records"]})
                connection.executemany(insert, batch)
                if count != result["records"]:
                    raise ValueError("Catalogue record count does not match its manifest.")
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
    metadata = {"id": dataset_id, "name": result["name"], "description": result.get("description", ""),
                "createdAt": now(), "storage": "sqlite", "recordCount": count, "catalogId": catalog_id,
                "license": result["license"], "source_url": result["source_url"], "revision": result["revision"],
                "manifestPath": result["manifest_path"], "split": result["split"], "sourceSha256": result["sha256"]}
    # ponytail: Workspace.apply does not exist yet; hold the lock just for the in-memory swap and save.
    with workspace.lock:
        workspace.state["datasets"] = [entry for entry in workspace.state["datasets"] if entry["id"] != dataset_id] + [metadata]
        workspace.save()
    registry_path = workspace.root / "data" / "catalog_registry.json"
    registered = [row for row in _read_state(registry_path, []) if isinstance(row, dict) and row.get("id") != dataset_id]
    atomic_json(registry_path, registered + [metadata])
    tombstones = workspace.root / "data" / "catalog_removed.json"
    if tombstones.exists():
        atomic_json(tombstones, [key for key in _read_state(tombstones, []) if key != dataset_id])
    return dataset_id


def reconcile_downloads(workspace, stop=None, emit=None):
    """Recover downloaded datasets missing from workspace metadata after an old app saved."""
    tombstones = workspace.root / "data" / "catalog_removed.json"
    removed = _read_state(tombstones, [])
    return [register_download(workspace, result, stop=stop, emit=emit) for result in list_downloads(workspace.root)
            if f"catalog_{result['catalog_id']}" not in removed]
