# sample.jsonl

The first 200 records of the GSM8K training split, used by `tests/e2e_train.py` and `tests/e2e_passive.py`.

- Source: https://github.com/openai/grade-school-math (`grade_school_math/data/train.jsonl`)
- Revision: `3101c7d5072418e28b9008a6636bde82a006892c` (file sha256 `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`)
- Licence: MIT. Copyright (c) 2021 OpenAI. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, subject to the inclusion of this copyright notice and the MIT licence text (https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE) in all copies or substantial portions of the Software. The Software is provided "as is", without warranty of any kind.

Each line is `{"user": question, "assistant": answer, "intent": "math_reasoning"}`, the field names the app's importer reads (the catalogue's `gsm` parser maps `question`/`answer` the same way).

Regenerate:

```python
import json, urllib.request
url = "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl"
rows = [json.loads(l) for l in urllib.request.urlopen(url).read().decode("utf-8").splitlines()][:200]
with open("sample.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        f.write(json.dumps({"user": r["question"], "assistant": r["answer"], "intent": "math_reasoning"}, ensure_ascii=False) + "\n")
```
