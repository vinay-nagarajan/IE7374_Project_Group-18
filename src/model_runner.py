from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import src...` / `import utils...` work whether invoked as
# `python src/model_runner.py` or `python -m src.model_runner`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import load_config, set_seed  # noqa: E402
from utils.helpers import (  # noqa: E402
    extract_evidence_spans,
    label_to_str,
    load_sample_notes,
    truncate_words,
    write_samples_jsonl,
    write_samples_txt,
)

DEFAULT_CONFIG = "configs/model_config.yaml"


# --------------------------------------------------------------- data loading

def _load_input_notes(cfg) -> tuple[list[dict], str]:
    """Return (notes, source_description).

    Prefers a real dataset.parquet from the M3 pipeline; falls back to the
    bundled synthetic notes so reviewers can always run the command.
    """
    n = int(cfg.data.n_samples)
    parquet_path = Path(cfg.data.dataset_parquet)

    if parquet_path.exists():
        try:
            import pandas as pd

            df = pd.read_parquet(parquet_path)
            # Prefer test-split rows if the column exists (most realistic).
            if "split" in df.columns and (df["split"] == "test").any():
                df = df[df["split"] == "test"]
            df = df.head(n)
            notes = []
            for i, row in df.reset_index(drop=True).iterrows():
                notes.append(
                    {
                        "note_id": f"MIMIC-{int(row.get('hadm_id', i))}",
                        "true_label": int(row.get("label", -1)),
                        # No trained classifier proba here; mark as unknown.
                        "predicted_label": int(row.get("label", -1)),
                        "predicted_proba": float("nan"),
                        "text": str(row["text"]),
                    }
                )
            return notes, f"real MIMIC notes ({parquet_path})"
        except Exception as e:  # pragma: no cover - defensive
            print(f"Could not read {parquet_path} ({e}); using synthetic notes.")

    raw = load_sample_notes(cfg.data.sample_notes)[:n]
    notes = [
        {
            "note_id": r["note_id"],
            "true_label": int(r["label"]),
            "predicted_label": int(r["predicted_label"]),
            "predicted_proba": float(r["predicted_proba"]),
            "text": r["text"],
        }
        for r in raw
    ]
    return notes, f"bundled synthetic notes ({cfg.data.sample_notes})"


# --------------------------------------------------------------- prompt build

def _build_prompt(note: dict, evidence: list[str], cfg) -> str:
    """Build a short, imperative prompt.

    Small instruction-tuned models (flan-t5-small) follow concise, direct
    prompts far better than long role-play framings, which they tend to copy.
    """
    pred = label_to_str(note["predicted_label"])
    evidence_text = " ".join(evidence)
    evidence_text = truncate_words(evidence_text, int(cfg.generation.max_input_words))
    return (
        f"Explain in 2 to 3 simple sentences why this discharge note indicates "
        f"{pred}. Base the explanation only on these findings: {evidence_text}"
    )


# Fragments we strip if a small model echoes the prompt back into its output.
_PROMPT_ECHO_MARKERS = (
    "you are a clinical assistant",
    "a model reviewed a hospital discharge",
    "and predicted:",
    "using only the evidence",
    "explain in 2 to 3 simple sentences",
    "base the explanation only on",
    "explanation:",
    "sentence 1:",
    "sentence 2:",
    "sentence 3:",
)


def _clean_generation(text: str, prompt: str) -> str:
    """Remove leaked prompt text and de-duplicate repeated sentences."""
    from utils.helpers import split_sentences  # local import to avoid cycle

    out = (text or "").strip()

    # Drop an exact prompt prefix if the model copied it verbatim.
    if out.lower().startswith(prompt.strip().lower()[:40]):
        out = out[len(prompt):].strip()

    # Drop known instruction fragments anywhere near the start.
    low = out.lower()
    for marker in _PROMPT_ECHO_MARKERS:
        idx = low.find(marker)
        if idx != -1 and idx < 120:
            out = out[idx + len(marker):].strip(" .:\n")
            low = out.lower()

    # De-duplicate repeated sentences (small-model repetition), ignoring case
    # and trailing punctuation so near-identical repeats collapse too.
    def _norm(s: str) -> str:
        return s.lower().strip().rstrip(".!?").strip()

    seen: list[str] = []
    seen_norms: set[str] = set()
    for sent in split_sentences(out):
        key = _norm(sent)
        if key and key not in seen_norms:
            seen.append(sent)
            seen_norms.add(key)
    return " ".join(seen).strip()


# --------------------------------------------------------------- generators

class _TemplateGenerator:
    """Deterministic fallback used when transformers/torch aren't installed.

    Produces a readable explanation from the evidence spans so the command
    still yields valid output. Clearly reported as the model in use.
    """

    name = "template-fallback (transformers not available)"

    def generate(self, note: dict, evidence: list[str], cfg) -> str:
        pred = label_to_str(note["predicted_label"])
        if not evidence:
            return (
                f"The note was assessed as {pred}, but no specific "
                "cue phrases were identified in the text."
            )
        # Compose a readable 2-3 sentence explanation from the evidence.
        findings = truncate_words(" ".join(evidence), 45)
        if int(note["predicted_label"]) == 1:
            return (
                f"This note points to {pred} because it documents findings such "
                f"as: {findings} Together these describe a pattern of progressive "
                "cognitive decline consistent with the prediction."
            )
        return (
            f"This note points to {pred} because it records: {findings} "
            "These statements indicate preserved cognition and no evidence of "
            "chronic memory impairment."
        )


class _HFGenerator:
    """Real pretrained generative model via HuggingFace transformers."""

    def __init__(self, cfg):
        from transformers import (  # lazy import
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

        model_name = cfg.generation.model_name
        self.model_type = str(cfg.generation.model_type)
        print(f"Loading generative model: {model_name} ({self.model_type}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.name = model_name

    def generate(self, note: dict, evidence: list[str], cfg) -> str:
        import torch

        prompt = _build_prompt(note, evidence, cfg)
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        gen_kwargs = dict(
            max_new_tokens=int(cfg.generation.max_new_tokens),
            min_new_tokens=int(getattr(cfg.generation, "min_new_tokens", 20)),
            num_beams=int(cfg.generation.num_beams),
            do_sample=bool(cfg.generation.do_sample),
            no_repeat_ngram_size=int(
                getattr(cfg.generation, "no_repeat_ngram_size", 3)
            ),
            repetition_penalty=float(
                getattr(cfg.generation, "repetition_penalty", 1.3)
            ),
            early_stopping=True,
        )
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        if self.model_type == "causal":
            # Strip the prompt tokens for causal LMs.
            text = self.tokenizer.decode(
                out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
            )
        else:
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return _clean_generation(text, prompt)


def _build_generator(cfg):
    """Return a generator, preferring the real model, falling back safely."""
    try:
        return _HFGenerator(cfg)
    except Exception as e:
        print(
            "NOTE: could not load the HuggingFace model "
            f"({type(e).__name__}: {e}).\n"
            "      Falling back to the deterministic template generator so the "
            "run still completes."
        )
        return _TemplateGenerator()


# --------------------------------------------------------------- orchestration

def run(config_path: str = DEFAULT_CONFIG) -> list[dict]:
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    outputs_dir = Path(cfg.paths.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading input notes")
    notes, source = _load_input_notes(cfg)
    print(f"      Source: {source}  ({len(notes)} notes)")

    print("\n[2/4] Loading generative model")
    generator = _build_generator(cfg)
    print(f"      Model in use: {generator.name}")

    print("\n[3/4] Generating explanations")
    records: list[dict] = []
    for i, note in enumerate(notes, 1):
        evidence = extract_evidence_spans(
            note["text"], max_spans=int(cfg.evidence.max_spans)
        )
        explanation = generator.generate(note, evidence, cfg)
        records.append(
            {
                "note_id": note["note_id"],
                "true_label": note["true_label"],
                "predicted_label": note["predicted_label"],
                "predicted_proba": note["predicted_proba"],
                "evidence": evidence,
                "explanation": explanation,
                "model": generator.name,
            }
        )
        print(f"      [{i}/{len(notes)}] {note['note_id']} -> done")

    print("\n[4/4] Saving outputs")
    write_samples_txt(records, cfg.paths.samples_txt)
    write_samples_jsonl(records, cfg.paths.samples_jsonl)
    print(f"      Wrote {cfg.paths.samples_txt}")
    print(f"      Wrote {cfg.paths.samples_jsonl}")
    print("\nDone. Generated explanations are in the outputs/ directory.")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate plain-language explanations of AD predictions (M4)"
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
