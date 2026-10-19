"""A failed queue job in plain words (#415): one sentence that says what happened and what to do next, with the raw text kept apart.

`explain(error)` takes the exception (in the worker, where its type is known) or just its text (a peer's report, a crash) and returns
`(sentence, details)`. `details` is the raw "Type: message" for a toggle and the log, and "" when the sentence is the error's own text
(an unknown error keeps it, and a message that already is a sentence is not repeated)."""
import errno
import json
import zipfile

START_AGAIN = "Start it again from the Queue page."

RAM = ("The computer ran out of memory (RAM) while this ran. Close other programs, or choose a smaller model, a shorter context or a "
       "smaller batch, then start it again.")
GPU = ("The graphics card ran out of memory while this ran. Lower the batch size or the context length, or close programs that use the "
       "GPU, then start it again.")
DISK = "The disk is full. Free space in Resources › Storage, then start it again."
CHECKPOINT = ("A saved checkpoint could not be used: it is damaged, or it was made with other settings. Remove this job and add it "
              "again from a version that trained, or from an earlier checkpoint.")
FOLDER = ("This run's folder already holds an earlier run, which was kept rather than overwritten. Remove the job and add it again to "
          "train into a new folder.")
DATA = ("The training data could not be read: a saved file is cut off or damaged. Import it again in Your data, then add this job "
        "again.")

# (needles in the lower-cased text, sentence): the first match wins. GPU memory before generic memory.
BY_TEXT = (
    (("cuda out of memory", "mps backend out of memory", "hip out of memory", "outofmemoryerror", "out of memory on device"), GPU),
    (("no space left on device", "not enough space on the disk", "disk full", "disk quota exceeded"), DISK),
    (("choose a new output folder",), FOLDER),
    (("memoryerror", "unable to allocate", "not enough memory", "cannot allocate memory", "out of memory"), RAM),
    (("unterminated string", "expecting value", "expecting property name", "extra data", "invalid control character",
      "expecting ',' delimiter", "jsondecodeerror", "codec can't decode", "unicodedecodeerror"), DATA),
    (("badzipfile", "not a zip file", "cannot load file containing pickled data", "failed to interpret file",
      "choose a groundwork numpy checkpoint", "checkpoint optimizer state", "resume must preserve", "resume requires the original",
      "this checkpoint has lora adapters", "eof when reading a line", "truncated"), CHECKPOINT),
)


def _by_type(error):
    if isinstance(error, MemoryError):
        return RAM
    if isinstance(error, OSError) and error.errno == errno.ENOSPC or getattr(error, "winerror", None) == 112:
        return DISK
    if isinstance(error, (json.JSONDecodeError, UnicodeDecodeError)):
        return DATA
    if isinstance(error, (zipfile.BadZipFile, EOFError)):
        return CHECKPOINT
    return None


def explain(error):
    """(sentence, details) for an exception or its text. An error nothing here knows keeps its own text."""
    text = str(error) or (type(error).__name__ if isinstance(error, BaseException) else "")
    raw = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) and str(error) else text
    sentence = _by_type(error) if isinstance(error, BaseException) else None
    if sentence is None:
        # the type's own name is part of what torch and numpy say (OutOfMemoryError), so search it too
        haystack = f"{type(error).__name__} {text}".lower() if isinstance(error, BaseException) else text.lower()
        sentence = next((s for needles, s in BY_TEXT if any(n in haystack for n in needles)), None)
    if sentence is None:
        return text, ""
    return sentence, raw
