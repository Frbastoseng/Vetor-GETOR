"""Dados de texto da v0.4 (pre-registrados em configs/v4_text_manifest.json).

4 dominios, documentos disjuntos entre train/val/final e de TUDO que foi usado nas versoes anteriores:
- en_wiki : WikiText-103 (train), excluindo todo artigo do WikiText-2 e do manifesto da v0.2.   CC BY-SA 3.0
- pt_wiki : Wikipedia PT (wikimedia/wikipedia, 20231101.pt, shard train-00003-of-00006).     CC BY-SA 4.0
- chat_en : HuggingFaceTB/smol-smoltalk (1o turno), excluindo ids da v0.3.                     Apache-2.0
- chat_pt : recogna-nlp/UltrachatBR (1o turno), excluindo ids da v0.3.                        MIT
Tambem separa prompts de usuario (nao usados em mais nada) para geracao livre e para a tarefa IF-v0.4.
O conjunto final so e lido com allow_final=True.
"""
from __future__ import annotations

import ast
import hashlib
import json

import numpy as np

from ..model_adapter import ROOT

MANIFEST = ROOT / "configs" / "v4_text_manifest.json"
CACHE = ROOT / "data" / "v4_texts.json"
SEED = 20261124
COUNTS = {"train": 300, "val": 80, "final": 100}
GEN_PROMPTS = {"chat_en": {"dev": 60, "final": 100}, "chat_pt": {"dev": 40, "final": 60}}
IF_PROMPTS = {"dev": 80, "final": 240}
PT_SHARD = "20231101.pt/train-00003-of-00006.parquet"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _v3_ids() -> set[str]:
    m = json.loads((ROOT / "configs" / "v3_train_manifest.json").read_text())
    return {e[0] for s in ("train", "val") for e in m[s]["ids"]}


def _en_wiki() -> list[tuple[str, str]]:
    from ..v2.data import _wt103_articles
    v2 = json.loads((ROOT / "configs" / "v2_splits_manifest.json").read_text())
    used = {e["id"].split(":", 1)[1] for s in v2["splits"].values() for e in s}
    arts = _wt103_articles()
    return [(f"wt103:{t}", arts[t]) for t in sorted(arts) if t not in used and len(arts[t]) > 2000]


def _pt_wiki() -> list[tuple[str, str]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("wikimedia/wikipedia", PT_SHARD, repo_type="dataset")
    df = pd.read_parquet(f, columns=["id", "title", "text"])
    return [(f"ptwiki:{i}", t) for i, t in zip(df["id"], df["text"]) if len(t) > 3000]


def _chats(tok, src: str, used: set[str]) -> tuple[list, list]:
    """Retorna (conversas formatadas [(id, texto)], prompts de usuario [(id, prompt)]), em ordem aleatoria fixa."""
    from datasets import load_dataset
    if src == "chat_en":
        ds, name = load_dataset("HuggingFaceTB/smol-smoltalk", split="train"), "smol-smoltalk"
    else:
        ds, name = load_dataset("recogna-nlp/UltrachatBR", split="train"), "UltrachatBR"
    rng = np.random.default_rng(SEED + (0 if src == "chat_en" else 1))
    convs, prompts = [], []
    need_c = sum(COUNTS.values())
    need_p = sum(GEN_PROMPTS[src].values()) + (sum(IF_PROMPTS.values()) if src == "chat_en" else 0)
    for i in rng.permutation(len(ds)):
        key = f"{name}:{int(i)}"
        if key in used:
            continue
        ex = ds[int(i)]
        if src == "chat_en":
            msgs = [{"role": m["role"], "content": m["content"]} for m in ex["messages"] if m["role"] in ("user", "assistant")][:2]
        else:
            try:
                t0 = ast.literal_eval(ex["conversa"])[0]
            except Exception:
                continue
            msgs = [{"role": "user", "content": t0.get("humano", "")}, {"role": "assistant", "content": t0.get("assistente", "")}]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or not msgs[1]["content"]:
            continue
        u = msgs[0]["content"].strip()
        if len(convs) < need_c and len(msgs[1]["content"]) > 1200:
            convs.append((key, tok.apply_chat_template(msgs, tokenize=False)))
        elif len(prompts) < need_p and 40 <= len(u) <= 400 and "\n" not in u:
            prompts.append((key, u))
        if len(convs) >= need_c and len(prompts) >= need_p:
            break
    return convs, prompts


def build_manifest(tok) -> dict:
    if MANIFEST.exists():
        raise FileExistsError("Manifesto v0.4 ja existe (pre-registro); nao recriar.")
    used = _v3_ids()
    texts, man = {}, {"seed": SEED, "counts": COUNTS, "pt_shard": PT_SHARD, "domains": {}, "prompts": {}}
    pools = {"en_wiki": _en_wiki(), "pt_wiki": _pt_wiki()}
    chat_prompts = {}
    for src in ("chat_en", "chat_pt"):
        pools[src], chat_prompts[src] = _chats(tok, src, used)
    for dom, pool in pools.items():
        if dom.endswith("wiki"):
            perm = np.random.default_rng(SEED + len(dom)).permutation(len(pool))
            pool = [pool[i] for i in perm]
        man["domains"][dom] = {}
        o = 0
        for split, n in COUNTS.items():
            rows = pool[o:o + n]
            o += n
            assert len(rows) == n, (dom, split, len(rows))
            man["domains"][dom][split] = [{"id": k, "sha": sha(t)} for k, t in rows]
            texts.update(dict(rows))
    for src, ps in chat_prompts.items():
        o = 0
        groups = dict(GEN_PROMPTS[src]) | ({f"if_{k}": v for k, v in IF_PROMPTS.items()} if src == "chat_en" else {})
        for g, n in groups.items():
            rows = ps[o:o + n]
            o += n
            assert len(rows) == n, (src, g, len(rows))
            man["prompts"][f"{src}:{g}"] = [{"id": k, "sha": sha(t)} for k, t in rows]
            texts.update({f"prompt|{k}": t for k, t in rows})
    ids = [e["id"] for d in man["domains"].values() for s in d.values() for e in s] + \
          [e["id"] for p in man["prompts"].values() for e in p]
    assert len(ids) == len(set(ids)), "sobreposicao entre conjuntos"
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    CACHE.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    return man


def load_texts(domain: str, split: str, *, allow_final: bool = False) -> list[dict]:
    if split == "final" and not allow_final:
        raise PermissionError("Conjunto final da v0.4 so com allow_final=True (avaliacao final).")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    texts = json.loads(CACHE.read_text(encoding="utf-8"))
    out = []
    for e in man["domains"][domain][split]:
        t = texts[e["id"]]
        if sha(t) != e["sha"]:
            raise ValueError(f"texto divergente do manifesto: {e['id']}")
        out.append({"doc_id": e["id"], "text": t})
    return out


def load_prompts(group: str, *, allow_final: bool = False) -> list[dict]:
    """group: 'chat_en:dev', 'chat_pt:final', 'chat_en:if_final', ..."""
    if group.endswith("final") and not allow_final:
        raise PermissionError("Prompts finais da v0.4 so com allow_final=True.")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    texts = json.loads(CACHE.read_text(encoding="utf-8"))
    out = []
    for e in man["prompts"][group]:
        t = texts[f"prompt|{e['id']}"]
        if sha(t) != e["sha"]:
            raise ValueError(f"prompt divergente do manifesto: {e['id']}")
        out.append({"id": e["id"], "prompt": t})
    return out
