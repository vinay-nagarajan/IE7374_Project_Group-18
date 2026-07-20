"""Extract frozen Bio_ClinicalBERT [CLS] embeddings, T4-safely.

The proposal uses Bio_ClinicalBERT as a *frozen* feature extractor: each note
-> its [CLS] token embedding (768-dim). No gradients, no fine-tuning.

T4 GPU safety (16 GB VRAM, and a free Colab host with ~13 GB RAM):
  * torch.no_grad() everywhere       -> no activation graph is stored
  * model.eval() + half precision    -> weights + activations ~2x smaller
  * small batch_size (16)            -> peak VRAM stays well under 16 GB
  * embeddings moved to CPU per batch -> GPU memory doesn't accumulate
  * torch.cuda.empty_cache() + gc     -> fragmentation released periodically
  * on-disk cache (.npz)             -> never recompute across runs/restarts

With these, extraction runs comfortably on a T4 and survives kernel restarts
(the CUDA/GPU issue you hit earlier) because finished work is cached to Drive.
"""

from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log_gpu(prefix: str = "") -> None:
    import torch

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"{prefix}GPU mem: {used:.2f} GB used / {reserved:.2f} GB reserved")


def extract_cls_embeddings(
    texts: list[str], cfg: SimpleNamespace, cache_path: str | None = None
) -> np.ndarray:
    """Return an (N, 768) float32 array of [CLS] embeddings.

    If `cache_path` exists and matches N, it is loaded instead of recomputed.
    """
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModel, AutoTokenizer

    # ---- disk cache ----
    if cache_path and Path(cache_path).exists():
        cached = np.load(cache_path)["emb"]
        if cached.shape[0] == len(texts):
            print(f"Loaded cached embeddings from {cache_path}: {cached.shape}")
            return cached
        print("Cache size mismatch -> recomputing embeddings.")

    device = _device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.embeddings.model_name)
    model = AutoModel.from_pretrained(cfg.embeddings.model_name)
    model.eval()
    model.to(device)

    use_fp16 = bool(cfg.embeddings.fp16) and device.type == "cuda"
    if use_fp16:
        model.half()  # weights -> fp16, halves VRAM footprint
    _log_gpu("After model load: ")

    bs = int(cfg.embeddings.batch_size)
    max_len = int(cfg.embeddings.max_length)
    all_emb: list[np.ndarray] = []

    for start in tqdm(range(0, len(texts), bs), desc="Embedding notes"):
        batch = texts[start : start + bs]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model(**enc)
            # [CLS] token is position 0 of last_hidden_state
            cls = out.last_hidden_state[:, 0, :]
            # back to fp32 on CPU for sklearn downstream
            all_emb.append(cls.float().cpu().numpy())

        # release per-batch GPU memory so it can't accumulate
        del enc, out, cls
        if device.type == "cuda" and (start // bs) % 25 == 0:
            torch.cuda.empty_cache()

    emb = np.vstack(all_emb).astype(np.float32)

    # free the model from GPU
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _log_gpu("After extraction + cleanup: ")

    print(f"Embeddings: {emb.shape}")
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, emb=emb)
        print(f"Cached embeddings -> {cache_path}")
    return emb
