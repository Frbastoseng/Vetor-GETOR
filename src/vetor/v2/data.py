"""Splits da v0.2 (pre-registrados em configs/v2_splits_manifest.json).

- final : 200 artigos do WikiText-103 (train) que NAO aparecem no WikiText-2. Avaliado UMA vez.
- val   : WikiText-2 validation + 140 artigos WT103 novos.
- train : WikiText-2 train + 1000 artigos WT103 novos.
- v01test: WikiText-2 test (ja conhecido pela v0.1) — so comparacao historica, nunca selecao.
Todos disjuntos por titulo de artigo. Licenca: CC BY-SA 3.0.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..collect import _TITLE, load_articles
from ..model_adapter import ROOT

MANIFEST = ROOT / "configs" / "v2_splits_manifest.json"
CACHE = ROOT / "data" / "v2_articles.json"
N_FINAL, N_VAL_EXTRA, N_TRAIN_EXTRA = 200, 140, 1000
SEED = 20261122


def _wt103_articles() -> dict[str, str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    docs, title, buf = {}, None, []
    for line in ds["text"]:
        s = line.rstrip("\n")
        if _TITLE.match(s):
            if title is not None and title not in docs:
                docs[title] = "".join(buf).strip()
            title, buf = s.strip(" ="), []
        elif title is not None:
            buf.append(line)
    if title is not None and title not in docs:
        docs[title] = "".join(buf).strip()
    return docs


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def build_manifest(cfg: dict) -> dict:
    """Cria (uma unica vez) o manifesto e o cache de textos. Falha se o manifesto ja existir."""
    if MANIFEST.exists():
        raise FileExistsError("Manifesto v0.2 ja existe; nao recriar (pre-registro).")
    wt2 = {s: load_articles(cfg, s) for s in ("train", "val", "test")}
    wt2_titles = {d["title"] for v in wt2.values() for d in v}
    w103 = _wt103_articles()
    pool = sorted(t for t, txt in w103.items() if t not in wt2_titles and len(txt) > 2000)
    perm = np.random.default_rng(SEED).permutation(len(pool))
    pick = [pool[i] for i in perm[: N_FINAL + N_VAL_EXTRA + N_TRAIN_EXTRA]]
    final = pick[:N_FINAL]
    val_x = pick[N_FINAL:N_FINAL + N_VAL_EXTRA]
    train_x = pick[N_FINAL + N_VAL_EXTRA:]
    arts = {}
    splits = {"final": [], "val": [], "train": [], "v01test": []}
    for t in final:
        arts[f"wt103:{t}"] = w103[t]; splits["final"].append(f"wt103:{t}")
    for t in val_x:
        arts[f"wt103:{t}"] = w103[t]; splits["val"].append(f"wt103:{t}")
    for t in train_x:
        arts[f"wt103:{t}"] = w103[t]; splits["train"].append(f"wt103:{t}")
    for s, key in (("train", "train"), ("val", "val"), ("test", "v01test")):
        for d in wt2[s]:
            k = f"wt2-{s}:{d['title']}"
            arts[k] = d["text"]; splits[key].append(k)
    man = {"seed": SEED, "source": "Salesforce/wikitext (wikitext-103-raw-v1 train, wikitext-2-raw-v1)",
           "license": "CC BY-SA 3.0",
           "rule": "final/val/train extras = permutacao com semente fixa dos artigos WT103-train fora do WT2 "
                   "(texto > 2000 caracteres); final avaliado uma unica vez",
           "counts": {k: len(v) for k, v in splits.items()},
           "splits": {k: [{"id": a, "sha": _sha(arts[a])} for a in v] for k, v in splits.items()}}
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(arts, ensure_ascii=False), encoding="utf-8")
    return man


def load_split(split: str, *, allow_final: bool = False) -> list[dict]:
    """Retorna [{doc_id, split, title, text}] verificando o hash de cada texto contra o manifesto."""
    if split == "final" and not allow_final:
        raise PermissionError("Conjunto final da v0.2 so pode ser lido com allow_final=True (avaliacao final).")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    arts = json.loads(CACHE.read_text(encoding="utf-8"))
    out = []
    for e in man["splits"][split]:
        txt = arts[e["id"]]
        if _sha(txt) != e["sha"]:
            raise ValueError(f"Texto divergente do manifesto: {e['id']}")
        out.append({"doc_id": e["id"], "split": split, "title": e["id"].split(":", 1)[1], "text": txt})
    return out
