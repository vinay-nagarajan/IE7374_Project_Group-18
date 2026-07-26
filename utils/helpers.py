from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_CUE_PATTERNS = [
    r"alzheimer'?s?", r"dementia", r"memory (?:loss|decline|impair\w*)",
    r"cognitive (?:decline|impair\w*)", r"confus\w+", r"disorient\w+",
    r"mmse", r"donepezil", r"memantine", r"neurocognitive", r"atrophy",
    r"word[- ]finding", r"repetitive question\w*", r"wandering",
]
_CUE_RE = re.compile("|".join(_CUE_PATTERNS), flags=re.IGNORECASE)

# Negative cues: phrases that argue AGAINST dementia. Surfacing these lets the
# generator explain control (no-dementia) predictions instead of returning only
# the boilerplate first sentence.
_NEG_PATTERNS = [
    r"cognition intact", r"no memory complaints?", r"alert and (?:fully )?oriented",
    r"no cognitive concerns?", r"returned to baseline", r"oriented throughout",
    r"no (?:prior )?history of dementia",
]
_NEG_RE = re.compile("|".join(_NEG_PATTERNS), flags=re.IGNORECASE)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Naive but dependency-free sentence splitter."""
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def extract_evidence_spans(text: str, max_spans: int = 4) -> list[str]:
    """Return the sentences most likely to have driven an AD prediction.

    Sentences are ranked by how many dementia-related cue phrases they contain.
    This gives the generative model a focused, relevant context instead of the
    full note, which improves the readability and faithfulness of explanations.
    """
    scored: list[tuple[int, str]] = []
    for sent in split_sentences(text):
        hits = len(_CUE_RE.findall(sent)) + len(_NEG_RE.findall(sent))
        if hits:
            scored.append((hits, sent))
    scored.sort(key=lambda x: x[0], reverse=True)
    spans = [s for _, s in scored[:max_spans]]
    # Fall back to the first sentence so we never hand the model an empty context.
    if not spans:
        sents = split_sentences(text)
        spans = sents[:1]
    return spans


def truncate_words(text: str, max_words: int) -> str:
    """Truncate to ``max_words`` words (keeps prompts within model limits)."""
    words = (text or "").split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def label_to_str(label: int) -> str:
    return "Alzheimer's disease / dementia" if int(label) == 1 else "no dementia"


# ---------------------------------------------------------------- io helpers

def load_sample_notes(path: str | Path) -> list[dict[str, Any]]:
    """Load the bundled synthetic notes used when MIMIC is unavailable."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sample notes not found at {path.resolve()}")
    with open(path, "r") as f:
        return json.load(f)


def write_samples_txt(records: list[dict[str, Any]], out_path: str | Path) -> None:
    """Write a human-readable transcript of every generated explanation."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("GENERATED EXPLANATIONS — Alzheimer's Detection (Milestone 4, RQ3)")
    lines.append("=" * 72)
    lines.append("")
    for i, r in enumerate(records, 1):
        lines.append(f"[Sample {i}] note_id={r['note_id']}")
        lines.append(f"  True label      : {label_to_str(r['true_label'])}")
        lines.append(
            f"  Model prediction: {label_to_str(r['predicted_label'])} "
            f"(p={r['predicted_proba']:.2f})"
        )
        lines.append(f"  Evidence spans  : {r['evidence']}")
        lines.append(f"  EXPLANATION     : {r['explanation']}")
        lines.append("-" * 72)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_samples_jsonl(records: list[dict[str, Any]], out_path: str | Path) -> None:
    """Write machine-readable JSONL (one explanation record per line)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
