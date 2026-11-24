"""v0.3 / Fase 2: dados de treino genericos (EN + PT-BR), descontaminados contra TODA a bateria de avaliacao.

Descontaminacao: remove conversas com qualquer 13-grama (palavras minusculas) presente em qualquer
exemplo de avaliacao (DEV e FINAL) das tarefas da bateria.
Saidas: data/v3_{train,val}.jsonl (nao versionado), configs/v3_train_manifest.json (versionado).
"""
from __future__ import annotations

import ast
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
import hashlib
import json
import re

import numpy as np

from vetor.model_adapter import ROOT, load_config
from vetor.v3.bench import SUITE

N_EN, N_PT, N_VAL = 12000, 4000, 600
MAX_TOKENS = 512
WORD = re.compile(r"\w+", re.UNICODE)


def ngrams(text: str, n: int = 13):
    w = WORD.findall(text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _strings(v)


def eval_ngrams() -> set[str]:
    from lm_eval.tasks import TaskManager, get_task_dict
    from vetor.v3.bench import leaf_tasks
    tm = TaskManager()
    grams = set()
    for t in SUITE:
        for _, task in leaf_tasks(get_task_dict([t], tm)).items():
            for doc in task.eval_docs:
                grams |= ngrams(" ".join(_strings(doc)))
    return grams


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(cfg["model"]["id"], revision=cfg["model"]["revision"])
    grams = eval_ngrams()
    print("13-gramas de avaliacao:", len(grams))
    rng = np.random.default_rng(20261123)
    out, stats = [], {"removidos_contaminacao": 0, "removidos_tamanho": 0}

    def accept(msgs, src, idx, lang):
        text = " ".join(m["content"] for m in msgs)
        if len(text) > 4 * MAX_TOKENS:  # filtro barato antes do caro
            stats["removidos_tamanho"] += 1
            return False
        ids = tok.apply_chat_template(msgs, tokenize=True)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        if len(ids) > MAX_TOKENS:
            stats["removidos_tamanho"] += 1
            return False
        if ngrams(text) & grams:
            stats["removidos_contaminacao"] += 1
            return False
        out.append({"id": f"{src}:{idx}", "lang": lang, "messages": msgs,
                    "sha": hashlib.sha256(json.dumps(msgs, ensure_ascii=False).encode()).hexdigest()[:16]})
        return True

    en = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    n = 0
    for i in rng.permutation(len(en)):
        ex = en[int(i)]
        msgs = [{"role": m["role"], "content": m["content"]} for m in ex["messages"] if m["role"] in ("user", "assistant")]
        if msgs and msgs[-1]["role"] == "assistant" and accept(msgs, "smol-smoltalk", int(i), "en"):
            n += 1
            if n >= N_EN + N_VAL // 2:
                break
    pt = load_dataset("recogna-nlp/UltrachatBR", split="train")
    n = 0
    for i in rng.permutation(len(pt)):
        try:
            conv = ast.literal_eval(pt[int(i)]["conversa"])
        except Exception:
            continue
        msgs = []
        for turn in conv[:1]:  # so o 1o turno (conversas do UltrachatBR sao longas)
            if "humano" in turn:
                msgs.append({"role": "user", "content": turn["humano"]})
            if "assistente" in turn:
                msgs.append({"role": "assistant", "content": turn["assistente"]})
        if msgs and msgs[-1]["role"] == "assistant" and accept(msgs, "UltrachatBR", int(i), "pt"):
            n += 1
            if n >= N_PT + N_VAL // 2:
                break
    en_rows = [r for r in out if r["lang"] == "en"]
    pt_rows = [r for r in out if r["lang"] == "pt"]
    val = en_rows[: N_VAL // 2] + pt_rows[: N_VAL // 2]
    train = en_rows[N_VAL // 2:] + pt_rows[N_VAL // 2:]
    for name, rows in (("train", train), ("val", val)):
        with open(ROOT / "data" / f"v3_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"fontes": {"en": "HuggingFaceTB/smol-smoltalk (Apache-2.0)", "pt": "recogna-nlp/UltrachatBR (MIT)"},
           "semente": 20261123, "max_tokens": MAX_TOKENS, "descontaminacao": "13-gramas vs todas as tarefas da bateria (DEV+FINAL)",
           "n_13gramas_avaliacao": len(grams), **stats,
           "train": {"n": len(train), "en": sum(r["lang"] == "en" for r in train), "ids": [(r["id"], r["sha"]) for r in train]},
           "val": {"n": len(val), "ids": [(r["id"], r["sha"]) for r in val]}}
    (ROOT / "configs" / "v3_train_manifest.json").write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
    print({k: v for k, v in man.items() if k not in ("train", "val")}, "train", man["train"]["n"], "val", man["val"]["n"])


if __name__ == "__main__":
    main()
