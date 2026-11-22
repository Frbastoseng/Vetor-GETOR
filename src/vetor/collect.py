"""Dados (split por ARTIGO) e professor de sensibilidade por ablacao isolada.

WikiText-2 raw (CC BY-SA 3.0). Usamos os splits oficiais train/validation/test, que ja
separam artigos inteiros; cada artigo e um documento. Nenhum prefixo cruza splits.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import torch
import torch.nn.functional as F

from .wrappers import RoutedMLP, reset, set_capture, stack_captured

_TITLE = re.compile(r"^ = [^=].* = $")
SPLIT_MAP = {"train": "train", "val": "validation", "test": "test"}


def load_articles(cfg: dict, split: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(cfg["data"]["dataset"], cfg["data"]["config"], split=SPLIT_MAP[split])
    docs, title, buf = [], None, []
    for line in ds["text"]:
        s = line.rstrip("\n")
        if _TITLE.match(s):
            if title is not None:
                docs.append({"title": title, "text": "".join(buf).strip()})
            title, buf = s.strip(" ="), []
        elif title is not None:
            buf.append(line)
    if title is not None:
        docs.append({"title": title, "text": "".join(buf).strip()})
    for i, d in enumerate(docs):
        d["doc_id"] = f"{split}:{i}:{hashlib.sha1(d['title'].encode()).hexdigest()[:8]}"
        d["split"] = split
    return docs


def tokenize_docs(tok, docs: list[dict], min_tokens: int) -> list[dict]:
    out = []
    for d in docs:
        ids = tok(d["text"], add_special_tokens=False).input_ids
        if len(ids) >= min_tokens:
            out.append({**d, "ids": ids})
    return out


def teacher_windows(docs: list[dict], per_doc: int, len_range: tuple[int, int], seed: int) -> list[dict]:
    """Janelas (prefixo, proximo token) nao sobrepostas dentro de cada documento."""
    rng = np.random.default_rng(seed)
    lo, hi = len_range
    out = []
    for d in docs:
        n = len(d["ids"])
        seg = n // per_doc
        for j in range(per_doc):
            t = int(rng.integers(lo, hi + 1))
            if seg < t + 1:
                continue
            start = j * seg + int(rng.integers(0, seg - t))
            ids = d["ids"][start:start + t + 1]
            out.append({"doc_id": d["doc_id"], "split": d["split"], "start": start,
                        "prefix": ids[:-1], "target": ids[-1]})
    return out


def eval_windows(docs: list[dict], prompt_len: int, cont_len: int, max_docs: int, seed: int) -> list[dict]:
    """Uma janela prompt+continuacao por documento (texto verdadeiro para teacher forcing)."""
    rng = np.random.default_rng(seed)
    need = prompt_len + cont_len
    pool = [d for d in docs if len(d["ids"]) >= need]
    order = rng.permutation(len(pool))[:max_docs]
    out = []
    for i in order:
        d = pool[int(i)]
        start = int(rng.integers(0, len(d["ids"]) - need + 1))
        ids = d["ids"][start:start + need]
        out.append({"doc_id": d["doc_id"], "split": d["split"], "start": start,
                    "prompt": ids[:prompt_len], "continuation": ids[prompt_len:]})
    return out


@torch.inference_mode()
def ablation_teacher(model, wraps: list[RoutedMLP], prefix: list[int] | torch.Tensor,
                     target: int | None, device: str) -> dict:
    """Um forward em lote [L+1, T]: linha 0 = FULL; linha l+1 = MLP da camada l zerado
    SO no ultimo token. Retorna z[l, ultimo] (da linha FULL), KL(p0||p_l) e DeltaNLL_l."""
    L = len(wraps)
    ids = torch.as_tensor(prefix, device=device).view(1, -1).expand(L + 1, -1).contiguous()
    reset(wraps)
    set_capture(wraps, True)
    for l, w in enumerate(wraps):
        w.ablate_rows = torch.tensor([l + 1], device=device)
    try:
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        Z = stack_captured(wraps, row=0).float()
    finally:
        reset(wraps)
    logp = F.log_softmax(logits, dim=-1)
    p0 = logp[0].exp()
    kl = (p0 * (logp[0] - logp[1:])).sum(-1).clamp_min(0)
    res = {"z": Z, "kl": kl}
    if target is not None:
        res["delta_nll"] = logp[0, target] - logp[1:, target]
        res["nll_full"] = -logp[0, target]
    return res
